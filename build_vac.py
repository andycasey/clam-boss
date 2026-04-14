import os
import numpy as np
import h5py
from astropy.table import Table
from tqdm import tqdm
import optax
import jax.numpy as jnp
from jax import jit
import jax.nn as jnn

from clam_boss.model import (
    compute_nmf_weights,
    infer_labels,
    build_design_matrix_batch_jax
)

from clam_boss.init_model import (
    load_ridge_model_npz,
    predict_with_ridge_npz
)


def infer_chunk(base_dir, flux, ivar, theta, H, label_mean, label_std, scatter):
    # predict from ridge model
    model_path = f'{base_dir}/nmf_ridge_model.npz'
    W_vals = compute_nmf_weights(flux, H)
    init_labels = predict_with_ridge_npz(
        W_vals,
        load_ridge_model_npz(model_path=model_path))
    init_labels_std = (init_labels - label_mean) / label_std

    # Infer labels using adam with decay schedule
    test_inferred_labels, test_label_covariances = infer_labels(
        flux, ivar,
        theta, H, label_mean, label_std, scatter,
        init_labels_std=init_labels_std,
        n_iter=1000,
        learning_rate=0.01,
        schedule=optax.cosine_decay_schedule(init_value=0.01, decay_steps=1000),
        optimizer='adam',
        grid_points=None,
        grid_range=None,
    )
    # refine for hot stars
    ev_hot = test_inferred_labels[:, 0] > 8000
    test_inferred_labels[ev_hot], test_label_covariances[ev_hot] = infer_labels(
            flux[ev_hot], ivar[ev_hot],
            theta, H, label_mean, label_std, scatter,
            init_labels_std=(test_inferred_labels[ev_hot] - label_mean) / label_std,
            n_iter=1000,
            learning_rate=0.01,
            optimizer='bfgs',
            grid_points=None,
            grid_range=None,
            logger=None,
            batch_size_bfgs=np.sum(ev_hot),
        )
    return test_inferred_labels, test_label_covariances


@jit
def spectra_rchi2(labels_std, theta_jnp, H_jnp, scatter_sq,
                  flux_jnp, var_jnp, n_wavelengths):
    design_matrix = build_design_matrix_batch_jax(labels_std)
    W = jnn.softplus(design_matrix @ theta_jnp)
    pred_flux = 1.0 - W @ H_jnp

    total_var = var_jnp + scatter_sq
    chi_sq = (flux_jnp - pred_flux)**2 / total_var
    rchi2 =  0.5 * jnp.sum(chi_sq, axis=1) / n_wavelengths
    return rchi2, pred_flux


if __name__ == '__main__':
    # load the block file
    block_file = '/data/stassun/medani/SDSS-V_data/DR20_data/mwmStarBlock-0.8.1.h5'
    bf = h5py.File(block_file, 'r')

    # load mwmAllStar to get the snr
    allstar_file = '/data/stassun/medani/SDSS-V_data/DR20_data/mwmAllStar-0.8.1.fits.gz'
    allstar = Table.read(allstar_file, hdu=1)

    # need to sort allstar
    keys1 = np.array(list(zip(bf['boss']['meta']['sdss_id'][:], [t.decode() for t in bf['boss']['meta']['telescope'][:]])), dtype=object)
    keys2 = np.array(list(zip(allstar['sdss_id'], allstar['telescope'])), dtype=object)

    # Build a lookup from key -> index in array 2
    lookup = {tuple(k): i for i, k in enumerate(keys2)}

    # Get the index in array 2 for each row in array 1
    idx2 = np.array([lookup[tuple(k)] for k in keys1])
    allstar = allstar[idx2]

    # only run on snr > 10 for now?
    idx_snr = np.where(allstar['snr'] > 10)[0]

    # load the params from saved model
    base_dir = 'model_results/nmf_joint_results_with_scatter_K32_alpha_m_w_wide_binaries_w_MS_w_HS'
    res = np.load(f'{base_dir}/joint_model_results.npz')
    theta = res['theta']
    H = res['H']
    label_mean = res['label_mean']
    label_std = res['label_std']
    scatter = res['scatter']

    theta_jnp = jnp.array(theta)
    H_jnp = jnp.array(H)
    scatter_sq = jnp.array(scatter)**2

    chunksize = 100000
    n_stars = len(idx_snr)
    n_batches = (n_stars + chunksize - 1) // chunksize
    n_wavelengths = bf['boss']['spectra']['flux'][:10].shape[1]

    save_dir = '/data/stassun/medani/SDSS-V_data/DR20_data'

    # create hdf5 file
    with h5py.File(f'{save_dir}/dr20_clam_inferred_parameters_w_rchi2.h5', "w") as f:
        f.create_dataset('inferred_labels',
                        shape=(n_stars, 4),
                        dtype='float64',
                        compression="gzip",
                        chunks=(25, 4))
        f.create_dataset('label_covariances',
                        shape=(n_stars, 4, 4),
                        dtype='float64',
                        compression="gzip",
                        chunks=(25, 4, 4))
        f.create_dataset('sdss_id',
                        shape=(n_stars,),
                        dtype=allstar['sdss_id'].dtype,
                        compression="gzip",
                        chunks=True)
        f.create_dataset('telescope',
                        shape=(n_stars,),
                        dtype=allstar['telescope'].dtype,
                        compression="gzip",
                        chunks=True)
        f.create_dataset('rchi2',
                        shape=(n_stars,),
                        dtype='float64',
                        compression="gzip",
                        chunks=True)
        f.create_dataset('model_flux',
                        shape=(n_stars, n_wavelengths),
                        dtype='float32',
                        compression="gzip",
                        chunks=(25, n_wavelengths))

    for i in tqdm(range(n_batches), desc="allstar batches"):
        start_idx = i * chunksize
        end_idx = min((i + 1) * chunksize, n_stars)
        idx_chunk = idx_snr[start_idx: end_idx]
        # dont run if already done the chunk
        if os.path.isfile(f'{save_dir}/dr20_clam_inferred_parameters_chunk_{i:03d}.npz'):
            chunk_res = np.load(f'{save_dir}/dr20_clam_inferred_parameters_chunk_{i:03d}.npz')
            inferred_labels_chunk = chunk_res['inferred_labels']
            label_covariances_chunk = chunk_res['label_covariances']
            rchi2_chunk = chunk_res['rchi2']
            model_flux_chunk = chunk_res['model_flux']

            with h5py.File(f'{save_dir}/dr20_clam_inferred_parameters_w_rchi2.h5', "a") as f:
                f['inferred_labels'][start_idx:end_idx] = inferred_labels_chunk
                f['label_covariances'][start_idx:end_idx] = label_covariances_chunk
                f['sdss_id'][start_idx:end_idx] = allstar['sdss_id'][idx_chunk]
                f['telescope'][start_idx:end_idx] = allstar['telescope'][idx_chunk]
                f['rchi2'][start_idx:end_idx] = rchi2_chunk
                f['model_flux'][start_idx:end_idx] = model_flux_chunk
        else:
            flux = bf['boss']['spectra']['flux'][idx_chunk]
            cont = bf['boss']['spectra']['continuum'][idx_chunk]
            ivar = bf['boss']['spectra']['ivar'][idx_chunk]

            norm_flux = flux / cont

            # Compute inverse variance for normalized flux
            norm_ivar = cont**2 * ivar

            # Handle non-finite values
            bad_pixels = (
                ~np.isfinite(norm_flux)
            |   ~np.isfinite(norm_ivar)
            |   (norm_ivar <= 0)
            |   (norm_flux <= 0)
            |   (norm_flux >= 1.2)
            )

            norm_flux = np.where(bad_pixels, 1.0, norm_flux)
            norm_ivar = np.where(bad_pixels, 0.0, norm_ivar)

            inferred_labels_chunk, label_covariances_chunk = infer_chunk(
                base_dir, norm_flux, norm_ivar, theta, H, label_mean, label_std, scatter)

            flux_jnp = jnp.array(norm_flux)
            var_jnp = 1.0 / jnp.maximum(norm_ivar, 1e-16)
            labels_std = jnp.array((inferred_labels_chunk - label_mean) / label_std)

            rchi2_res, pred_flux_res = spectra_rchi2(
                labels_std, theta_jnp, H_jnp, scatter_sq,
                flux_jnp, var_jnp, n_wavelengths)
            rchi2_chunk = np.array(rchi2_res)
            model_flux_chunk = np.array(pred_flux_res, dtype='float32')

            with h5py.File(f'{save_dir}/dr20_clam_inferred_parameters_w_rchi2.h5', "a") as f:
                f['inferred_labels'][start_idx:end_idx] = inferred_labels_chunk
                f['label_covariances'][start_idx:end_idx] = label_covariances_chunk
                f['sdss_id'][start_idx:end_idx] = allstar['sdss_id'][idx_chunk]
                f['telescope'][start_idx:end_idx] = allstar['telescope'][idx_chunk]
                f['rchi2'][start_idx:end_idx] = rchi2_chunk
                f['model_flux'][start_idx:end_idx] = model_flux_chunk
            
            # save the chunk just in case
            np.savez_compressed(f'{save_dir}/dr20_clam_inferred_parameters_chunk_{i:03d}.npz',
                                inferred_labels=inferred_labels_chunk,
                                label_covariances=label_covariances_chunk,
                                sdss_id=allstar['sdss_id'][idx_chunk],
                                telescope=allstar['telescope'][idx_chunk],
                                rchi2=rchi2_chunk,
                                model_flux=model_flux_chunk)
    
    # clean up the chunks
    for i in tqdm(range(n_batches), desc="deleting chunk files"):
        os.remove(f'{save_dir}/dr20_clam_inferred_parameters_chunk_{i:03d}.npz')
