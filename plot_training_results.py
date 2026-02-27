import numpy as np
import matplotlib.pylab as plt
from matplotlib.colors import LogNorm
import os
import configparser
from nmf_stellar_model_joint import (load_data, plot_test_comparison,
                                     kiel_diagram, alpha_fe_plot)


if __name__ == '__main__':
    # Configuration
    config = configparser.ConfigParser()
    base_dir = 'model_results/nmf_joint_results_with_scatter_K32_alpha_m_w_wide_binaries_w_MS_w_HS'
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

    # load the data like in original script
     # run code
    if add_WBs:
        append_wb = '_w_wide_binaries'
        data_file_wb = 'boss_apogee_wide_binary_training_data.npz'
    else:
        append_wb = ''

    if add_MS:
        append_ms = '_w_MS'
        data_file_ms = 'boss_minesweeper_training_data.npz'
    else:
        append_ms = ''

    if add_HS:
        append_hs = '_w_HS'
        data_file_hs = 'boss_hot_star_training_data.npz'
    else:
        append_hs = ''

    if convert_alpha:
        label_names = ['teff', 'logg', 'm_h', 'alpha_h']
        output_dir = f'model_results/nmf_joint_results_with_scatter_K32{append_wb}{append_ms}{append_hs}'
    else:
        label_names = ['teff', 'logg', 'm_h', 'alpha_m']
        output_dir = f'model_results/nmf_joint_results_with_scatter_K32_alpha_m{append_wb}{append_ms}{append_hs}'

    # Load data
    data_file = 'boss_apogee_lux_training_data.npz'
    absorption, flux, ivar, true_labels = load_data(data_file,
                                                    convert_alpha=convert_alpha)
    data_type = np.array(['nominal'] * len(flux))
    if add_MS:
        absorption_ms, flux_ms, ivar_ms, true_labels_ms = load_data(
            data_file_ms,
            convert_alpha=convert_alpha)
        # apply offsets found in Vedant's paper
        true_labels_ms[:, 2] -= -0.05
        true_labels_ms[:, 3] -= 0.04
        # append the MS stars
        absorption = np.append(absorption, absorption_ms, axis=0)
        flux = np.append(flux, flux_ms, axis=0)
        ivar = np.append(ivar, ivar_ms, axis=0)
        true_labels = np.append(true_labels, true_labels_ms, axis=0)
        data_type = np.append(data_type, ['minesweeper'] * len(flux_ms), axis=0)

    if add_WBs:
        absorption_wb, flux_wb, ivar_wb, true_labels_wb = load_data(
            data_file_wb,
            convert_alpha=convert_alpha)
    if remove_nans:  # combine data now if not doing EM
        # append the WBs
        absorption = np.append(absorption, absorption_wb, axis=0)
        flux = np.append(flux, flux_wb, axis=0)
        ivar = np.append(ivar, ivar_wb, axis=0)
        true_labels = np.append(true_labels, true_labels_wb, axis=0)
        data_type = np.append(data_type, ['wide_binaries'] * len(flux_wb), axis=0)

        # remove nans
        labels_mask = np.isfinite(true_labels)
        keep_stars = np.all(labels_mask, axis=1)
        absorption = absorption[keep_stars]
        flux = flux[keep_stars]
        ivar = ivar[keep_stars]
        true_labels = true_labels[keep_stars]
        data_type = data_type[keep_stars]
    
    if add_HS:
        # load HSs, append later
        absorption_hs, flux_hs, ivar_hs, true_labels_hs = load_data(
            data_file_hs,
            convert_alpha=convert_alpha)

        absorption = np.append(absorption, absorption_hs, axis=0)
        flux = np.append(flux, flux_hs, axis=0)
        ivar = np.append(ivar, ivar_hs, axis=0)
        true_labels = np.append(true_labels, true_labels_hs, axis=0)
        data_type = np.append(data_type, ['hot_stars'] * len(flux_hs), axis=0)

    # load the saved labels
    res = np.load(f'{output_dir}/test_inference_results.npz')
    test_inferred_labels = res['test_inferred_labels']

    # now do each plot individually for each type
    data_types = ['nominal', 'minesweeper', 'wide_binaries', 'hot_stars']
    for dt in data_types:
        ev = (data_type == dt)
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
