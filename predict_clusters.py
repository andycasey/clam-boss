import os
import numpy as np
import matplotlib.pyplot as plt
from nmf_stellar_model_joint import (infer_labels, plot_spectra_comparison,
                                     predict_with_ridge_npz, load_ridge_model_npz,
                                     compute_nmf_weights)
import warnings
import matplotlib.colors as colors
import matplotlib.cm as cm
import configparser


def load_data(file_path, open_clusters):
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
    if open_clusters:
        labels = np.column_stack([
            data['cluster_feh'],
            data['pm_prob'],
            data['rv_prob'],
            data['feh_prob']
        ])
    else:
        labels = np.column_stack([
            data['cluster_feh'],
            data['rv_prob'],
            data['vb_prob']
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

    bad_vals = abs(test_inferred_labels[:, 2]) > 10

    if cluster_feh == -9999.:
        cluster_feh0 = cluster_feh
        cluster_feh = np.nanmean(test_inferred_labels[:, 2][~bad_vals])
        plot_mean = False
    else:
        cluster_feh0 = cluster_feh
        plot_mean = True
    maxx_diff = np.nanpercentile(abs(test_inferred_labels[:, 2][~bad_vals] - cluster_feh), 95)

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

    ax2.hist(test_inferred_labels[:, 2][~bad_vals])
    if plot_mean:
        ax2.axvline(cluster_feh, linestyle='--', c='r')
    ax2.set_xlabel('[Fe/H]')
    ax2.set_ylabel('N')
    ax2.grid()
    
    mean = np.nanmean(test_inferred_labels[:, 2][~bad_vals])
    std = np.nanstd(test_inferred_labels[:, 2][~bad_vals])
    plt.suptitle(f"{cluster}: [Fe/H] = {cluster_feh0:.4f}, CLAM [Fe/H] = {mean:.4f} +/- {std:.4f}")
    plt.savefig(f"{output_dir}/{cluster}.png", dpi=200)
    plt.close()


def plot_compare_clusers(mean_fe_h_plot,
                         clusters_plot,
                         test_inferred_labels_all,
                         cluster_all,
                         output_dir,
                         feh_color=True):
    """
    plot multiple clusters same plot to compare spreads
    """
    if feh_color:
        norm = colors.Normalize(
            vmin=mean_fe_h_plot.min() - 0.1,
            vmax=mean_fe_h_plot.max() + 0.1
        )

        cmap = cm.inferno
    else:
        norm = colors.Normalize(
            vmin=0,
            vmax=len(mean_fe_h_plot)
        )

        cmap = cm.tab10

    f, ax1 = plt.subplots(1, 1, figsize=(12, 10))
    for i in range(len(mean_fe_h_plot)):
        ev = (cluster_all == clusters_plot[i])
        if feh_color:
            color = cmap(norm(mean_fe_h_plot[i]))
        else:
            color = cmap(norm(i))
        ax1.scatter(
            test_inferred_labels_all[ev, 0],
            test_inferred_labels_all[ev, 1],
            color=color,
            label=clusters_plot[i]
        )
    ax1.invert_xaxis()
    ax1.invert_yaxis()
    ax1.set_xlabel('Teff')
    ax1.set_ylabel('log(g)')
    ax1.grid()
    ax1.legend(ncols=2)
    
    if feh_color:
        # add colorbar
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax1)
        cbar.set_label('Cluster Median [Fe/H] (CLAM)')
        plt.savefig(f"{output_dir}/compare_cluster_kiels_feh_color.png", dpi=200)
    else:
        plt.savefig(f"{output_dir}/compare_cluster_kiels.png", dpi=200)
    plt.close()



if __name__ == '__main__':
    # Configuration
    config = configparser.ConfigParser()
    base_dir = 'nmf_joint_results_with_scatter_K32_alpha_m_w_wide_binaries_w_MS_w_HS'
    default_cfg = f'{base_dir}/default.cfg'
    config.read(default_cfg)

    K = config.getint('settings', 'K')
    n_iter = config.getint('settings', 'n_iter')
    learning_rate = config.getfloat('settings', 'learning_rate')
    print_every = config.getint('settings', 'print_every')
    convert_alpha = config.getboolean('settings', 'convert_alpha')
    add_WBs = config.getboolean('settings', 'add_WBs')
    add_MS = config.getboolean('settings', 'add_MS')
    add_HS = config.getboolean('settings', 'add_HS')
    remove_nans = config.getboolean('settings', 'remove_nans')
    train_w_subsample = config.getboolean('settings', 'train_w_subsample')

    # run code
    if add_WBs:
        append_wb = '_w_wide_binaries'
    else:
        append_wb = ''

    if add_MS:
        append_ms = '_w_MS'
    else:
        append_ms = ''

    if add_HS:
        append_hs = '_w_HS'
    else:
        append_hs = ''

    # do open clusters first
    open_clusters = True
    if open_clusters:
        data_file = 'boss_cluster_stars_data.npz'
        dir_start = 'boss_cluster_validation'
    else:
        data_file = 'boss_gc_stars_data.npz'
        dir_start = 'boss_gc_validation'

    if convert_alpha:
        label_names = ['teff', 'logg', 'm_h', 'alpha_h']
        save_dir = base_dir
        output_dir = f'{dir_start}_{append_wb}{append_ms}{append_hs}'
    else:
        label_names = ['teff', 'logg', 'm_h', 'alpha_m']
        save_dir = base_dir
        output_dir = f'{dir_start}_alpha_m{append_wb}{append_ms}{append_hs}'
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving results to: {output_dir}/")

    # Load data
    print("\nLoading data...")
    absorption, flux, ivar, true_labels, cluster = load_data(data_file, open_clusters)
    n_stars, n_wavelengths = flux.shape
    print(f"  Loaded {n_stars} stars with {n_wavelengths} wavelength pixels")

    # load the params from saved model
    res = np.load(f'{save_dir}/joint_model_results.npz')
    theta = res['theta']
    H = res['H']
    label_mean = res['label_mean']
    label_std = res['label_std']
    scatter = res['scatter']

    # predict from ridge model
    model_path = f'{save_dir}/nmf_ridge_model.npz'
    W_vals = compute_nmf_weights(flux, H)
    init_labels = predict_with_ridge_npz(
        W_vals,
        load_ridge_model_npz(model_path=model_path))
    init_labels_std = (init_labels - label_mean) / label_std

    # Infer labels using BFGS with grid search initialization
    test_inferred_labels = infer_labels(
        flux, ivar,
        theta, H, label_mean, label_std, scatter,
        init_labels_std=init_labels_std,
        n_iter=[100, 1000],
        learning_rate=0.01,
        optimizer='two-stage',
        grid_points=None,
        grid_range=None
    )

    # Wavelength grid
    loglam = 3.5523 + 0.0001 * np.arange(n_wavelengths)
    wavelength = 10**loglam

    cluster_unq, cluster_counts = np.unique(cluster, return_counts=True)
    mean_feh = np.zeros(len(cluster_unq))

    for i, clust in enumerate(cluster_unq):
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
        mean_feh[i] = np.nanmedian(test_inferred_labels[idx][:, 2])

    # plot all OCs on one plot
    clusters_ignore = ['Alessi_20', 'Berkeley_59', 'NGC_457']
    clusters_plot = []
    mean_fe_h_plot = []
    for i in range(len(cluster_unq)):
        if cluster_unq[i] not in clusters_ignore:
            clusters_plot.append(cluster_unq[i])
            mean_fe_h_plot.append(mean_feh[i])
    clusters_plot = np.array(clusters_plot)
    mean_fe_h_plot = np.array(mean_fe_h_plot)
    idsort = np.argsort(mean_fe_h_plot)
    clusters_plot = clusters_plot[idsort]
    mean_fe_h_plot = mean_fe_h_plot[idsort]

    # plot the results
    plot_compare_clusers(mean_fe_h_plot,
                         clusters_plot,
                         test_inferred_labels,
                         cluster,
                         output_dir,
                         feh_color=True)
    plot_compare_clusers(mean_fe_h_plot,
                         clusters_plot,
                         test_inferred_labels,
                         cluster,
                         output_dir,
                         feh_color=False)

    # save in new arrays
    test_inferred_labels_all = test_inferred_labels.copy()
    cluster_all = cluster.copy()
    cluster_unq_all = cluster_unq.copy()
    cluster_counts_all = cluster_counts.copy()
    mean_feh_all = mean_feh.copy()

    # do gcs
    open_clusters = False
    if open_clusters:
        data_file = 'boss_cluster_stars_data.npz'
        dir_start = 'boss_cluster_validation'
    else:
        data_file = 'boss_gc_stars_data.npz'
        dir_start = 'boss_gc_validation'

    if convert_alpha:
        label_names = ['teff', 'logg', 'm_h', 'alpha_h']
        save_dir = base_dir
        output_dir = f'{dir_start}_{append_wb}{append_ms}{append_hs}'
    else:
        label_names = ['teff', 'logg', 'm_h', 'alpha_m']
        save_dir = base_dir
        output_dir = f'{dir_start}_alpha_m{append_wb}{append_ms}{append_hs}'
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving results to: {output_dir}/")

    # Load data
    print("\nLoading data...")
    absorption, flux, ivar, true_labels, cluster = load_data(data_file, open_clusters)
    n_stars, n_wavelengths = flux.shape
    print(f"  Loaded {n_stars} stars with {n_wavelengths} wavelength pixels")

    # load the params from saved model
    res = np.load(f'{save_dir}/joint_model_results.npz')
    theta = res['theta']
    H = res['H']
    label_mean = res['label_mean']
    label_std = res['label_std']
    scatter = res['scatter']

    # predict from ridge model
    model_path = f'{save_dir}/nmf_ridge_model.npz'
    W_vals = compute_nmf_weights(flux, H)
    init_labels = predict_with_ridge_npz(
        W_vals,
        load_ridge_model_npz(model_path=model_path))
    init_labels_std = (init_labels - label_mean) / label_std

    # Infer labels using BFGS with grid search initialization
    test_inferred_labels = infer_labels(
        flux, ivar,
        theta, H, label_mean, label_std, scatter,
        init_labels_std=init_labels_std,
        n_iter=[100, 1000],
        learning_rate=0.01,
        optimizer='two-stage',
        grid_points=None,
        grid_range=None
    )

    # Wavelength grid
    loglam = 3.5523 + 0.0001 * np.arange(n_wavelengths)
    wavelength = 10**loglam

    cluster_unq, cluster_counts = np.unique(cluster, return_counts=True)
    mean_feh = np.zeros(len(cluster_unq))

    for i, clust in enumerate(cluster_unq):
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
        mean_feh[i] = np.nanmedian(test_inferred_labels[idx][:, 2])

    # append all array
    test_inferred_labels_all = np.append(test_inferred_labels_all,
                                         test_inferred_labels,
                                         axis=0)
    cluster_all = np.append(cluster_all, cluster)
    cluster_unq_all = np.append(cluster_unq_all, cluster_unq)
    cluster_counts_all = np.append(cluster_counts_all, cluster_counts)
    mean_feh_all = np.append(mean_feh_all, mean_feh)

    # pick those to plot on there own plot
    feh_bins = np.linspace(mean_feh_all.min(),
                           mean_feh_all.max() + 0.01,
                           6)
    clusters_ignore = ['Alessi_20', 'Berkeley_59']
    clusters_plot = ['NGC6205', 'NGC5904', 'NGC0104', 'NGC_2682', 'NGC_6791']
    mean_fe_h_plot = []
    if len(clusters_plot) == 0:
        for i in range(len(feh_bins) - 1):
            feh_ev = (mean_feh_all >= feh_bins[i]) & \
                    (mean_feh_all < feh_bins[i + 1]) & \
                    (~np.isin(cluster_unq_all, clusters_ignore))
            mean_fehi = mean_feh_all[feh_ev]
            clusti = cluster_unq_all[feh_ev]
            countsi = cluster_counts_all[feh_ev]
            clusters_plot.append(clusti[np.argmax(countsi)])
            mean_fe_h_plot.append(mean_fehi[np.argmax(countsi)])
    else:
        for i in range(len(clusters_plot)):
            mean_fe_h_plot.append(mean_feh_all[cluster_unq_all == clusters_plot[i]][0])

    # plot the results
    plot_compare_clusers(np.array(mean_fe_h_plot),
                         clusters_plot,
                         test_inferred_labels_all,
                         cluster_all,
                         output_dir,
                         feh_color=True)
    plot_compare_clusers(np.array(mean_fe_h_plot),
                         clusters_plot,
                         test_inferred_labels_all,
                         cluster_all,
                         output_dir,
                         feh_color=False)
