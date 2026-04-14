"""
Joint NMF-based stellar spectra model.

This script jointly optimizes:
1. Stellar labels for all training stars
2. Polynomial coefficients mapping labels -> NMF weights
3. NMF basis vectors (H)

The objective is to minimize spectral reconstruction error while ensuring:
- Predicted weights W = design_matrix(labels) @ theta are non-negative
- H basis vectors are non-negative
"""


import os
import shutil
import logging
import configparser
import numpy as np
import optax

from clam_boss.utils import (
    load_data,
    compute_label_statistics,
    unf_training_sample,
)

from clam_boss.model import (
    joint_optimization,
    compute_nmf_weights,
    infer_labels,
)

from clam_boss.init_model import (
    train_and_save_ridge_model,
    load_ridge_model_npz,
    predict_with_ridge_npz
)

from clam_boss.plot import (
    plot_test_comparison,
    plot_comparison,
    plot_nmf_components,
    plot_loss,
    plot_model_scatter,
    plot_spectra_comparison,
    plot_residual_histograms,
    kiel_diagram,
    alpha_fe_plot,
)


if __name__ == '__main__':
    # Configuration
    config = configparser.ConfigParser()
    default_cfg = 'default.cfg'
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

    if convert_alpha:
        label_names = ['teff', 'logg', 'm_h', 'alpha_h']
        output_dir = f'model_results/nmf_joint_results_with_scatter_K32{append_wb}{append_ms}{append_hs}'
    else:
        label_names = ['teff', 'logg', 'm_h', 'alpha_m']
        output_dir = f'model_results/nmf_joint_results_with_scatter_K32_alpha_m{append_wb}{append_ms}{append_hs}'

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    # save current cfg file
    shutil.copy(default_cfg, output_dir)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{output_dir}/optimization.log'),
            logging.StreamHandler()  # Also print to console
        ]
    )
    logger = logging.getLogger(__name__)

    logger.info(f"Saving results to: {output_dir}/")

    logger.info("=" * 60)
    logger.info("Joint NMF Stellar Spectra Model")
    logger.info("=" * 60)
    logger.info(f"Configuration:")
    logger.info(f"  K = {K} components")
    logger.info(f"  Iterations = {n_iter}")
    logger.info(f"  Learning rate = {learning_rate}")
    logger.info(f"  Converting Alpha = {convert_alpha}")
    logger.info(f"  Adding Wide Binaries = {add_WBs}")
    logger.info(f"  Adding Mineweeper = {add_MS}")
    logger.info(f"  Adding Hot Stars = {add_HS}")
    logger.info(f"  Training set subsampled = {train_w_subsample}")
    logger.info(f"  Removing naned labels = {remove_nans}")

    # Load data
    logger.info("\n[1/4] Loading data...")
    data_file = 'boss_clam_training_data.npz'
    absorption, flux, ivar, true_labels, meta_data = load_data(
        data_file,
        convert_alpha=convert_alpha)
    keep = (meta_data[:, -1] == 'nominal')

    if add_MS:
        keep += (meta_data[:, -1] == 'minesweeper')
        true_labels[meta_data[:, -1] == 'minesweeper', 2] -= -0.05  # apply offsets found in Vedant's paper
        true_labels[meta_data[:, -1] == 'minesweeper', 3] -= 0.097  # subtract off median diff Peter found
    if add_WBs:
        keep += (meta_data[:, -1] == 'wide_binary')
    if add_HS:
        # load HSs, append later
        hs = meta_data[:, -1] == 'hot_stars'
        absorption_hs = absorption[hs]
        flux_hs = flux[hs]
        ivar_hs = ivar[hs]
        true_labels_hs = true_labels[hs]
        meta_data_hs = meta_data[hs]
    absorption = absorption[keep]
    flux = flux[keep]
    ivar = ivar[keep]
    true_labels = true_labels[keep]
    meta_data = meta_data[keep]
    if remove_nans:  # combine data now if not doing EM
        # remove nans
        labels_mask = np.isfinite(true_labels)
        keep_stars = np.all(labels_mask, axis=1)
        absorption = absorption[keep_stars]
        flux = flux[keep_stars]
        ivar = ivar[keep_stars]
        true_labels = true_labels[keep_stars]
        meta_data = meta_data[keep_stars]
    
    n_stars, n_wavelengths = flux.shape
    logger.info(f"  Loaded {n_stars} stars with {n_wavelengths} wavelength pixels")

    # Joint optimization
    logger.info("\n[2/4] Running joint optimization...")
    # get training subsample
    if train_w_subsample:
        if add_MS:
            nbins = 40
            nstars_per_bin = 7
            ranges = [[2800, 6500],
                      [1, 5.25],
                      [-0.1, 0.5]]
            bins_use = [0, 1, 3]
        elif add_WBs and remove_nans:
            nbins = 40
            nstars_per_bin = 7
            ranges = [[2800, 6500],
                      [1, 5.25],
                      [-0.1, 0.5]]
            bins_use = [0, 1, 3]
        else:
            nbins = 15
            nstars_per_bin = 7
            ranges = [[4000, 6500],
                      [1, 5],
                      [-2, 0.5],
                      [-0.1, 0.3]]
            bins_use = [0, 1, 2, 3]
        idx_train = unf_training_sample(
            true_labels[:, bins_use],
            nbins, nstars_per_bin,
            random_seed=42, ranges=ranges)
    else:
        idx_train = np.arange(len(true_labels))

    if add_HS:
        # add in the hot stars at end
        absorption = np.append(absorption, absorption_hs, axis=0)
        flux = np.append(flux, flux_hs, axis=0)
        ivar = np.append(ivar, ivar_hs, axis=0)
        true_labels = np.append(true_labels, true_labels_hs, axis=0)
        meta_data = np.append(meta_data, meta_data_hs, axis=0)

        # add some WBs to training set
        rng = np.random.default_rng(seed=42)
        id_bins = np.arange(n_stars, len(flux), 1)
        idx_train_bin = rng.choice(
            id_bins,
            size=int(len(id_bins) * 0.8), replace=False)
        
        idx_train = np.append(idx_train, idx_train_bin)

    labels_mask = np.isfinite(true_labels)
    init_stars = np.all(labels_mask, axis=1)
    per_label_weights = np.zeros_like(true_labels) + 1.
    per_label_weights[~labels_mask] = 0.0

    inferred_labels, theta, H, W, label_mean, label_std, losses, scatter = joint_optimization(
        flux[idx_train], ivar[idx_train], true_labels[idx_train], K,
        n_iter=n_iter,
        learning_rate=learning_rate,
        print_every=print_every,
        seed=42,
        per_label_weights=per_label_weights[idx_train],
        logger=logger
    )

    # Compute statistics
    logger.info("\n[3/4] Computing statistics...")
    stats = compute_label_statistics(true_labels[idx_train], inferred_labels, label_names)

    logger.info("\n" + "=" * 60)
    logger.info("Summary Statistics (Training Set)")
    logger.info("=" * 60)
    for name in label_names:
        s = stats[name]
        logger.info(f"  {name:8s}: bias={s['bias']:+.4f}, scatter={s['scatter']:.4f}, MAD={s['mad']:.4f}")

    # Generate plots
    logger.info("\n[4/4] Generating plots...")

    # Wavelength grid
    loglam = 3.5523 + 0.0001 * np.arange(n_wavelengths)
    wavelength = 10**loglam

    plot_comparison(true_labels[idx_train], inferred_labels, label_names, f'{output_dir}/label_comparison.png')
    plot_residual_histograms(true_labels[idx_train], inferred_labels, label_names, f'{output_dir}/label_residuals.png')
    plot_nmf_components(H, f'{output_dir}/nmf_components.png')
    if len(losses) > 1:
        plot_loss(losses, f'{output_dir}/optimization_loss.png')
    plot_spectra_comparison(
        flux[idx_train], ivar[idx_train], true_labels[idx_train], inferred_labels,
        theta, H, label_mean, label_std, wavelength,
        f'{output_dir}/spectra_comparison.png', n_plot=20
    )
    plot_model_scatter(
        wavelength, scatter,
        f'{output_dir}/model_scatter.png'
    )

    # Save results
    logger.info("\nSaving results...")
    np.savez(f'{output_dir}/joint_model_results.npz',
             inferred_labels=inferred_labels,
             true_labels=true_labels,
             meta_data=meta_data,
             idx_train=idx_train,
             theta=theta,
             H=H,
             W=W,
             label_mean=label_mean,
             label_std=label_std,
             losses=losses,
             scatter=scatter,
             stats=stats,
             convert_alpha=convert_alpha)

    logger.info(f"  Saved to {output_dir}/")

    # =========================================================================
    # TEST STEP: Infer labels from spectra alone (no known labels)
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("Test Step: Inferring labels from spectra (no known labels)")
    logger.info("=" * 60)

    # Use all data as "test" - pretend we don't know the labels
    # In practice you'd use a held-out test set
    test_flux = flux
    test_ivar = ivar
    test_true_labels = true_labels

    logger.info(f"Test set: {len(test_flux)} spectra")
    logger.info("Inferring labels using trained model (theta, H fixed)...")

    logger.info('"Training Ridge Regression to use as initial guesser...')

    # train model
    # if WBs, replace nan log(g) with infered
    labels_ridge = true_labels[idx_train].copy()
    if add_WBs:
        mask = labels_mask[idx_train]
        labels_ridge[~mask] = inferred_labels[~mask]
    W_train = compute_nmf_weights(flux[idx_train], H)
    model_path = f'{output_dir}/nmf_ridge_model.npz'
    train_and_save_ridge_model(W_train, labels_ridge,
                               save_path=model_path, alpha=1.,
                               logger=logger)
    # predict from this model
    W_train = compute_nmf_weights(test_flux, H)
    init_labels = predict_with_ridge_npz(W_train, load_ridge_model_npz(model_path=model_path))
    init_labels_std = (init_labels - label_mean) / label_std

    logger.info("\n" + "=" * 60)
    logger.info("Summary Statistics (Ridge Regression)")
    logger.info("=" * 60)
    stats = compute_label_statistics(test_true_labels, init_labels, label_names)
    for name in label_names:
        s = stats[name]
        logger.info(f"  {name:8s}: bias={s['bias']:+.4f}, scatter={s['scatter']:.4f}, MAD={s['mad']:.4f}")


    # create percentile based gridding
    # true_norm = (true_labels - label_mean) / label_std
    # grid_points = [40, 40, 22, 10]  # do denser in teff, logg
    # grid_range = [np.nanpercentile(true_norm[:, i], [0.1, 99.9]) for i in range(true_norm.shape[1])]

    # Infer labels using only flux and ivar
    test_inferred_labels, test_label_covariances = infer_labels(
        test_flux, test_ivar,
        theta, H, label_mean, label_std, scatter,
        init_labels_std=init_labels_std,
        n_iter=1000,
        learning_rate=0.01,
        schedule=optax.cosine_decay_schedule(init_value=0.01, decay_steps=1000),
        optimizer='adam',
        grid_points=None,
        grid_range=None,
        logger=logger
    )

    # refine for hot stars
    ev_hot = test_inferred_labels[:, 0] > 8000
    test_inferred_labels[ev_hot], test_label_covariances[ev_hot] = infer_labels(
            test_flux[ev_hot], test_ivar[ev_hot],
            theta, H, label_mean, label_std, scatter,
            init_labels_std=(test_inferred_labels[ev_hot] - label_mean) / label_std,
            n_iter=1000,
            learning_rate=0.01,
            optimizer='bfgs',
            grid_points=None,
            grid_range=None,
            logger=logger,
            batch_size_bfgs=np.sum(ev_hot),
        )

    # Compute test statistics
    test_stats = compute_label_statistics(test_true_labels, test_inferred_labels, label_names)

    logger.info("\n" + "=" * 60)
    logger.info("Test Set Statistics")
    logger.info("=" * 60)
    for name in label_names:
        s = test_stats[name]
        logger.info(f"  {name:8s}: bias={s['bias']:+.4f}, scatter={s['scatter']:.4f}, MAD={s['mad']:.4f}")

    # Plot test results: true vs inferred
    logger.info("\nGenerating test comparison plot...")
    plot_test_comparison(
        test_true_labels, test_inferred_labels, label_names,
        f'{output_dir}/test_true_vs_inferred.png'
    )

    # kiel diagram
    kiel_diagram(test_true_labels,
                 test_inferred_labels,
                 label_names,
                 output_dir,
                 fe_h=False)

    # keil diagram with Fe/H
    kiel_diagram(test_true_labels,
                 test_inferred_labels,
                 label_names,
                 output_dir,
                 fe_h=True)

    # alpha/M vs Fe/H
    alpha_fe_plot(test_true_labels,
                  test_inferred_labels,
                  label_names,
                  output_dir,
                  convert_alpha)

    # now do each plot individually for each type
    data_types = ['nominal', 'minesweeper', 'wide_binary', 'hot_stars']
    for dt in data_types:
        ev = (meta_data[:, -1] == dt)
        if np.sum(ev) > 0:
            ranges = []
            for i in range(true_labels.shape[1]):
                minn, maxx = np.nanpercentile(true_labels[ev, i], [0, 100])
                if not np.isnan(minn):
                    minn *= 0.95
                else:
                    minn = -1.
                if not np.isnan(maxx):
                    maxx *= 1.05
                else:
                    maxx = 1.
                ranges.append((minn, maxx))
            plot_test_comparison(
                true_labels[ev], test_inferred_labels[ev], label_names,
                f'{output_dir}/test_true_vs_inferred_{dt}.png',
                label_bounds={
                                'teff': ranges[0],
                                'logg': ranges[1],
                                'm_h': ranges[2],
                                'alpha_m': ranges[3]
                            }
            )

            fe_h_perc = np.nanpercentile(true_labels[ev, 2], 1)
            if np.isnan(fe_h_perc):
                fe_h_perc = -1

            # kiel diagram
            kiel_diagram(true_labels[ev],
                         test_inferred_labels[ev],
                         label_names,
                         output_dir,
                         fe_h=False,
                         name_append=f'_{dt}',
                         teff_max=ranges[0][1])

            # keil diagram with Fe/H
            kiel_diagram(true_labels[ev],
                         test_inferred_labels[ev],
                         label_names,
                         output_dir,
                         fe_h=True,
                         name_append=f'_{dt}',
                         teff_max=ranges[0][1],
                         feh_min=fe_h_perc)

            # alpha/M vs Fe/H
            if dt != 'hot_stars':
                alpha_fe_plot(true_labels[ev],
                              test_inferred_labels[ev],
                              label_names,
                              output_dir,
                              convert_alpha,
                              name_append=f'_{dt}',
                              feh_min=fe_h_perc)

    # Save test results
    np.savez(f'{output_dir}/test_inference_results.npz',
             test_true_labels=test_true_labels,
             test_inferred_labels=test_inferred_labels,
             test_label_covariances=test_label_covariances,
             meta_data=meta_data,
             test_stats=test_stats)
    logger.info(f"Saved test results to {output_dir}/test_inference_results.npz")

    logger.info("\nDone!")
