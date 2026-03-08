import os
import numpy as np
import h5py
from astropy.table import Table
from tqdm import tqdm
import optax
import jax.numpy as jnp

from clam_boss.model import (
    compute_nmf_weights,
    load_ridge_model_npz,
    predict_with_ridge_npz,
    infer_labels,
    build_design_matrix_batch_jax,
)

@jit
def spectra_rchi2(labels_std, theta_jnp, H_jnp, scatter_sq,
                  flux_jnp, var_jnp, n_wavelengths):
    design_matrix = build_design_matrix_batch_jax(labels_std)
    W = jnn.softplus(design_matrix @ theta_jnp)
    pred_flux = 1.0 - W @ H_jnp

    total_var = var_jnp + scatter_sq
    chi_sq = (flux_jnp - pred_flux)**2 / total_var
    rchi2 =  0.5 * jnp.sum(chi_sq, axis=1) / n_wavelengths
    return rchi2



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

    params = np.load(f'{save_dir}/dr20_clam_inferred_parameters.npz')
    inferred_labels = params['inferred_labels']
    label_covariances = params['label_covariances']
    rchi2 = np.zeros(n_stars)

    save_dir = '/data/stassun/medani/SDSS-V_data/DR20_data'

    for i in tqdm(range(n_batches), desc="allstar batches"):
        start_idx = i * chunksize
        end_idx = min((i + 1) * chunksize, n_stars)
        idx_chunk = idx_snr[start_idx: end_idx]

        flux = bf['boss']['spectra']['flux'][idx_chunk]
        cont = bf['boss']['spectra']['continuum'][idx_chunk]
        ivar = bf['boss']['spectra']['ivar'][idx_chunk]

        n_wavelengths = flux.shape[1]

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

        flux_jnp = jnp.array(norm_flux)
        var_jnp = 1.0 / jnp.maximum(norm_ivar, 1e-16)
        labels_std = jnp.array((inferred_labels[start_idx: end_idx] - label_mean) / label_std)

        rchi2_res = spectra_rchi2(
            labels_std, theta_jnp, H_jnp, scatter_sq,
            flux_jnp, var_jnp, n_wavelengths)
        rchi2[start_idx: end_idx] = np.array(rchi2_res)

    # save all the results
    np.savez_compressed(f'{save_dir}/dr20_clam_inferred_parameters_w_rchi2.npz',
                        inferred_labels=inferred_labels,
                        label_covariances=label_covariances,
                        rchi2=rchi2,
                        sdss_id=allstar['sdss_id'][idx_snr],
                        telescope=allstar['telescope'][idx_snr])
