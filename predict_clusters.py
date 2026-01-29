import os
import numpy as np
import matplotlib.pyplot as plt
from nmf_stellar_model_joint import infer_labels, plot_spectra_comparison
import warnings


def load_data(file_path):
    """Load and preprocess the training data."""
    data = np.load(file_path)['boss_cluster_data']

    # Normalize flux
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        norm_flux = data['flux'] / data['continuum']

    # Compute inverse variance for normalized flux
    norm_ivar = data['continuum']**2 * data['ivar']

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

    # Compute absorption: A = 1 - flux, clipped to be non-negative
    absorption = np.clip(1.0 - norm_flux, 0.0, np.inf)

    # Extract stellar labels: teff, logg, m_h, alpha_h
    labels = np.column_stack([
        data['cluster_feh'],
        data['pm_prob'],
        data['rv_prob'],
        data['feh_prob']
    ])
    cluster = data['cluster']

    return absorption, norm_flux, norm_ivar, labels, cluster


def plot_cluster_results(test_inferred_labels,
                         cluster,
                         cluster_feh,
                         output_dir):
    """
    plot the cluster results
    """
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))

    if cluster_feh == -9999.:
        cluster_feh0 = cluster_feh
        cluster_feh = np.nanmean(test_inferred_labels[:, 2])
        plot_mean = False
    else:
        cluster_feh0 = cluster_feh
        plot_mean = True
    maxx_diff = np.nanpercentile(abs(test_inferred_labels[:, 2] - cluster_feh), 95)

    dens = ax1.scatter(test_inferred_labels[:, 0],
                       test_inferred_labels[:, 1],
                       c=test_inferred_labels[:, 2],
                       vmin=cluster_feh - maxx_diff,
                       vmax=cluster_feh + maxx_diff,
                       cmap='seismic')
    plt.colorbar(dens, ax=ax1, label='[Fe/H]')
    ax1.invert_xaxis()
    ax1.invert_yaxis()
    ax1.set_xlabel('Teff')
    ax1.set_ylabel('log(g)')
    ax1.grid()

    ax2.hist(test_inferred_labels[:, 2])
    if plot_mean:
        ax2.axvline(cluster_feh, linestyle='--', c='r')
    ax2.set_xlabel('[Fe/H]')
    ax2.set_ylabel('N')
    ax2.grid()
    
    mean = np.nanmean(test_inferred_labels[:, 2])
    std = np.nanstd(test_inferred_labels[:, 2])
    plt.suptitle(f"{cluster}: [Fe/H] = {cluster_feh0:.4f}, CLAM [Fe/H] = {mean:.4f} +/- {std:.4f}")
    plt.savefig(f"{output_dir}/{cluster}.png", dpi=200)
    plt.close()


if __name__ == '__main__':
    # Configuration
    data_file = 'boss_cluster_stars_data.npz'
    K = 32
    n_iter = 10_000
    learning_rate = 0.01 # 0.1 is too aggressive
    print_every = 1000

    save_dir = 'nmf_joint_results_with_scatter_K32'
    label_names = ['teff', 'logg', 'm_h', 'alpha_h']

    # Create output directory
    output_dir = 'boss_cluster_validation'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving results to: {output_dir}/")

    # Load data
    print("\nLoading data...")
    absorption, flux, ivar, true_labels, cluster = load_data(data_file)
    n_stars, n_wavelengths = flux.shape
    print(f"  Loaded {n_stars} stars with {n_wavelengths} wavelength pixels")

    # load the params from saved model
    res = np.load(f'{save_dir}/joint_model_results.npz')
    theta = res['theta']
    H = res['H']
    label_mean = res['label_mean']
    label_std = res['label_std']
    scatter = res['scatter']
    # Infer labels using BFGS with grid search initialization
    test_inferred_labels = infer_labels(
        flux, ivar,
        theta, H, label_mean, label_std, scatter,
        init_labels_std=None,  # triggers grid search
        n_iter=1000,
        optimizer='bfgs',
        grid_points=5
    )

    # Wavelength grid
    loglam = 3.5523 + 0.0001 * np.arange(n_wavelengths)
    wavelength = 10**loglam

    cluster_unq = np.unique(cluster)
    

    for clust in cluster_unq:
        idx = np.where(cluster == clust)[0]
        plot_cluster_results(test_inferred_labels[idx],
                             cluster[idx][0],
                             true_labels[idx[0], 0],
                             output_dir)

        plot_spectra_comparison(
        flux[idx], ivar[idx], test_inferred_labels[idx], test_inferred_labels[idx],
        theta, H, label_mean, label_std, wavelength,
        f'{output_dir}/{clust}_spectra_comparison.png', n_plot=20
    )
