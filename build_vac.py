import os
import numpy as np
import h5py
from astropy.table import Table
from tqdm import tqdm

from clam_boss.model import (
    compute_nmf_weights,
    load_ridge_model_npz,
    predict_with_ridge_npz,
    infer_labels,
)


def infer_chunk(base_dir, flux, ivar, theta, H, label_mean, label_std, scatter):
    # predict from ridge model
    model_path = f'{base_dir}/nmf_ridge_model.npz'
    W_vals = compute_nmf_weights(flux, H)
    init_labels = predict_with_ridge_npz(
        W_vals,
        load_ridge_model_npz(model_path=model_path))
    init_labels_std = (init_labels - label_mean) / label_std

    # Infer labels using BFGS with grid search initialization
    test_inferred_labels, test_label_covariances = infer_labels(
        flux, ivar,
        theta, H, label_mean, label_std, scatter,
        init_labels_std=init_labels_std,
        n_iter=[100, 1000],
        learning_rate=0.01,
        optimizer='two-stage',
        grid_points=None,
        grid_range=None,
        batch_size_bfgs=500,
    )
    return test_inferred_labels, test_label_covariances



if __name__ == '__main__':
    # load the block file
    block_file = '/uufs/chpc.utah.edu/common/home/sdss51/sdsswork/mwm/spectro/astra/0.8.1/spectra/block/mwmStarBlock-0.8.1.h5'
    bf = h5py.File(block_file, 'r')

    # load mwmAllStar to get the snr
    allstar_file = '/uufs/chpc.utah.edu/common/home/sdss51/sdsswork/mwm/spectro/astra/0.8.1/summary/mwmAllStar-0.8.1.fits.gz'
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

    chunksize = 10000
    n_stars = len(idx_snr)
    n_batches = (n_stars + chunksize - 1) // chunksize

    inferred_labels = np.zeros((n_stars, 4))
    label_covariances = np.zeros((n_stars, 4, 4))

    for i in tqdm(range(n_batches), desc="allstar batches"):
        start_idx = i * chunksize
        end_idx = min((i + 1) * chunksize, n_stars)

        idx_chunk = idx_snr[start_idx: end_idx]

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

        inferred_labels[start_idx: end_idx], label_covariances[start_idx: end_idx] = infer_chunk(
            base_dir, norm_flux, norm_ivar, theta, H, label_mean, label_std, scatter)

    # save the results
    np.savez_compressed('clam_inferred_parameters.npz',
                        inferred_labels=inferred_labels,
                        label_covariances=label_covariances,
                        sdss_id=allstar['sdss_id'][idx_snr],
                        telescope=allstar['telescope'][idx_snr])
