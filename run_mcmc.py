import os
import numpy as np
import h5py
from astropy.table import Table
from tqdm import tqdm
import optax
import jax.numpy as jnp
from jax import jit
import jax.nn as jnn

from clam_boss.mcmc import mcmc_params


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

    save_dir = '/data/stassun/medani/SDSS-V_data/DR20_data'
    save_hdf5_file = 'dr20_clam_inferred_parameters_w_rchi2_new_init.h5'
    params = h5py.File(f'{save_dir}/{save_hdf5_file}', "r")

    # load the set we are going to run the MCMC on
    mcmc_samples = Table.read('mcmc_sample.csv')
    # telescope column was written as str(bytes) (e.g. "b'apo25m'") instead of
    # the decoded string, so strip that back off before matching
    mcmc_samples['telescope'] = [t.strip("b'\"") for t in mcmc_samples['telescope']]

    def h5_fancy_index(dataset, idx):
        """Gather rows of an h5py dataset at arbitrary (unsorted) integer
        positions. h5py only allows strictly increasing, duplicate-free
        fancy-index arrays, so sort, fetch, then undo the sort."""
        order = np.argsort(idx)
        gathered = dataset[idx[order]]
        undo = np.argsort(order)
        return gathered[undo]

    # (sdss_id, telescope) -> row index, for the spectra block file
    bf_telescope = np.array([t.decode() for t in bf['boss']['meta']['telescope'][:]])
    bf_lookup = {k: i for i, k in enumerate(
        zip(bf['boss']['meta']['sdss_id'][:], bf_telescope))}

    # (sdss_id, telescope) -> row index, for the inferred-labels file
    # (params has a different length/order than bf, so this needs its own lookup)
    params_telescope = np.array([t.decode() for t in params['telescope'][:]])
    params_lookup = {k: i for i, k in enumerate(
        zip(params['sdss_id'][:], params_telescope))}

    sample_keys = list(zip(mcmc_samples['sdss_id'], mcmc_samples['telescope']))
    idx_bf = np.array([bf_lookup[k] for k in sample_keys])
    idx_params = np.array([params_lookup[k] for k in sample_keys])

    chunksize = 1000
    n_stars = len(mcmc_samples)
    n_batches = (n_stars + chunksize - 1) // chunksize
    batch_files = []
    for i in tqdm(range(n_batches), desc="mcmc batches"):
        start_idx = i * chunksize
        end_idx = min((i + 1) * chunksize, n_stars)

        batch_file = f'dr20_mcmc_samples_{start_idx}_{end_idx}.npz'
        if os.path.isfile(batch_file):
            batch_files.append(batch_file)
            continue

        idx_bf_chunk = idx_bf[start_idx: end_idx]
        idx_params_chunk = idx_params[start_idx: end_idx]

        flux = h5_fancy_index(bf['boss']['spectra']['flux'], idx_bf_chunk)
        cont = h5_fancy_index(bf['boss']['spectra']['continuum'], idx_bf_chunk)
        ivar = h5_fancy_index(bf['boss']['spectra']['ivar'], idx_bf_chunk)

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

        best_labels = h5_fancy_index(params['inferred_labels'], idx_params_chunk)
        label_covariances = h5_fancy_index(params['label_covariances'], idx_params_chunk)
        label_sigma = np.sqrt(np.diagonal(label_covariances, axis1=1, axis2=2))

        default_sigma = np.array([275., 0.3, 0.15, 0.11])
        label_sigma = np.where(~np.isfinite(label_sigma), default_sigma, label_sigma)

        samples = mcmc_params(
            norm_flux,
            norm_ivar,
            best_labels,
            theta,
            H,
            label_mean,
            label_std,
            scatter,
            default_sigma)

        np.savez(batch_file, samples=samples)
        batch_files.append(batch_file)

    # combine all batches into one array (rows stay in mcmc_samples order),
    # save as a single file, and remove the per-batch files
    all_samples = np.concatenate(
        [np.load(f)['samples'] for f in batch_files], axis=0)
    np.savez('dr20_mcmc_samples_all.npz', samples=all_samples)

    for f in batch_files:
        os.remove(f)
