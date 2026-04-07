import numpy as np
import jax.numpy as jnp
import jax.nn as jnn
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import configparser
import jax
from clam_boss.model import build_design_matrix_np

jax.config.update("jax_enable_x64", True)


def plot_test_comparison(true_labels, inferred_labels,
                         label_names, save_path,
                         label_bounds={
                            'teff': (2500, 20000),
                            'logg': (0.5, 5.5),
                            'm_h': (-4., 0.75),
                            'alpha_h': (-0.5, 0.6)
                         },
                         logger=None):
    """Create comparison plots of true vs inferred labels for test set."""
    n_labels = true_labels.shape[1]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()

    for i, (ax, name) in enumerate(zip(axes, label_names)):
        true_vals = true_labels[:, i]
        inferred_vals = inferred_labels[:, i]

        diff = inferred_vals - true_vals
        bias = np.nanmedian(diff)
        scatter = np.nanstd(diff)
        mad = np.nanmedian(np.abs(diff - bias))

        ax.scatter(true_vals, inferred_vals, alpha=0.5, s=10, c='steelblue', edgecolors='none')

        bounds = label_bounds.get(name, (np.nanmin(true_vals), np.nanmax(true_vals)))
        ax.plot(bounds, bounds, 'r-', lw=2, label='1:1')

        ax.set_xlabel(f'True {name}', fontsize=12)
        ax.set_ylabel(f'Inferred {name}', fontsize=12)
        ax.set_xlim(bounds)
        ax.set_ylim(bounds)
        ax.set_title(f'{name} (Test Set)\nbias={bias:.3f}, scatter={scatter:.3f}, MAD={mad:.3f}', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.legend(loc='upper left')

    plt.suptitle('Test Set: True vs Inferred Labels', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    if logger is not None:
        logger.info(f"Saved test comparison plot to {save_path}")


def plot_comparison(true_labels, inferred_labels,
                    label_names, save_path,
                    label_bounds = {
                        'teff': (2500, 20000),
                        'logg': (0.5, 5.5),
                        'm_h': (-4., 0.75),
                        'alpha_h': (-0.5, 0.6)
                    },
                    logger=None):
    """Create comparison plots of true vs inferred labels."""
    n_labels = true_labels.shape[1]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()

    for i, (ax, name) in enumerate(zip(axes, label_names)):
        true_vals = true_labels[:, i]
        inferred_vals = inferred_labels[:, i]

        diff = inferred_vals - true_vals
        bias = np.nanmedian(diff)
        scatter = np.nanstd(diff)
        mad = np.nanmedian(np.abs(diff - bias))

        ax.scatter(true_vals, inferred_vals, alpha=0.3, s=2, c='black')

        bounds = label_bounds.get(name, (np.nanmin(true_vals), np.nanmax(true_vals)))
        ax.plot(bounds, bounds, 'r-', lw=2, label='1:1')

        ax.set_xlabel(f'True {name}')
        ax.set_ylabel(f'Inferred {name}')
        ax.set_xlim(bounds)
        ax.set_ylim(bounds)
        ax.set_title(f'{name}\nbias={bias:.3f}, scatter={scatter:.3f}, MAD={mad:.3f}')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    if logger is not None:
        logger.info(f"Saved comparison plot to {save_path}")


def plot_nmf_components(H, save_path, logger=None):
    """Plot the NMF spectral components."""
    K = H.shape[0]
    n_wavelengths = H.shape[1]

    loglam = 3.5523 + 0.0001 * np.arange(n_wavelengths)
    wavelength = 10**loglam

    n_cols = int(np.ceil(np.sqrt(K)))
    n_rows = int(np.ceil(K / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 2*n_rows), sharex=True)
    axes = axes.ravel()

    for k in range(K):
        ax = axes[k]
        ax.plot(wavelength, H[k], 'k-', lw=0.5)
        ax.set_ylabel(f'{k+1}', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, H[k].max() * 1.1)

    for k in range(K, len(axes)):
        axes[k].set_visible(False)

    plt.suptitle('NMF Spectral Components (Absorption Basis)', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    if logger is not None:
        logger.info(f"Saved NMF components plot to {save_path}")


def plot_loss(losses, save_path, logger=None):
    """Plot optimization loss."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(losses)
    ax1.set_xlabel('Checkpoint')
    ax1.set_ylabel('Loss')
    ax1.set_title('Joint Optimization Loss')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)

    n_last = min([50, len(losses)])
    ax2.plot(losses[-n_last:])
    ax2.set_xlabel(f'Checkpoint (last {n_last})')
    ax2.set_ylabel('Loss')
    ax2.set_title(f'Loss (Last {n_last} checkpoints)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    if logger is not None:
        logger.info(f"Saved loss plot to {save_path}")


def plot_model_scatter(wavelength, scatter, save_path, logger=None):
    """
        f'{output_dir}/model_scatter.png'
    )
    """

    fig, ax = plt.subplots()
    ax.plot(wavelength, scatter, 'k-', lw=1.0)
    ax.set_xlabel('Wavelength (A)')
    ax.set_ylabel('Model Scatter')
    ax.set_ylim(0, 0.1)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close("all")
    if logger is not None:
        logger.info(f"Saved model scatter plot to {save_path}")



def plot_spectra_comparison(flux, ivar, true_labels, inferred_labels,
                            theta, H, label_mean, label_std, wavelength,
                            save_path, n_plot=20):
    """Plot observed vs model spectra."""
    n_subset = min([n_plot, len(flux)])

    # Random selection
    np.random.seed(123)
    indices = np.random.choice(len(flux), size=n_subset, replace=False)

    n_cols = 2
    n_rows = (n_subset + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows), sharex=True)
    axes = axes.ravel()

    for idx, i in enumerate(indices):
        ax = axes[idx]

        obs_flux = flux[i]

        # Model flux from inferred labels
        inf_labels_std = (inferred_labels[i] - label_mean) / label_std
        design_vec = build_design_matrix_np(inf_labels_std.reshape(1, -1))
        inf_weights = np.array(jnn.softplus(design_vec @ theta))[0]
        model_flux = 1.0 - inf_weights @ H

        # Model flux from true labels
        true_labels_std = (true_labels[i] - label_mean) / label_std
        design_vec_true = build_design_matrix_np(true_labels_std.reshape(1, -1))
        true_weights = np.array(jnn.softplus(design_vec_true @ theta))[0]
        true_model_flux = 1.0 - true_weights @ H

        ax.plot(wavelength, obs_flux, 'k-', lw=0.5, alpha=0.7, label='Observed')
        ax.plot(wavelength, model_flux, 'r-', lw=0.8, alpha=0.8, label='Model (inferred)')
        ax.plot(wavelength, true_model_flux, 'b--', lw=0.8, alpha=0.6, label='Model (true labels)')

        true_lbl = true_labels[i]
        inf_lbl = inferred_labels[i]
        ax.set_title(f'Star {i+1}: Teff={true_lbl[0]:.0f}->{inf_lbl[0]:.0f}, '
                     f'logg={true_lbl[1]:.2f}->{inf_lbl[1]:.2f}, '
                     f'[M/H]={true_lbl[2]:.2f}->{inf_lbl[2]:.2f}', fontsize=9)
        ax.set_ylim(0.4, 1.1)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=8)

    for idx in range(n_subset, len(axes)):
        axes[idx].set_visible(False)

    if len(axes) >= 2:
        axes[-2].set_xlabel('Wavelength (A)')
        axes[-1].set_xlabel('Wavelength (A)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_residual_histograms(true_labels, inferred_labels,
                             label_names, save_path, logger=None):
    """Plot histograms of residuals."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    for i, (ax, name) in enumerate(zip(axes, label_names)):
        residuals = inferred_labels[:, i] - true_labels[:, i]
        residuals = residuals[np.isfinite(residuals)]

        ax.hist(residuals, bins=100, color='steelblue', edgecolor='black', alpha=0.7)
        ax.axvline(0, color='red', linestyle='--', lw=2)

        median = np.median(residuals)
        mad = np.median(np.abs(residuals - median))

        ax.axvline(median, color='orange', linestyle='-', lw=2, label=f'Median={median:.3f}')

        ax.set_xlabel(f'Inferred - True {name}')
        ax.set_ylabel('Count')
        ax.set_title(f'{name} Residuals (MAD={mad:.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    if logger is not None:
        logger.info(f"Saved residual histogram to {save_path}")


def kiel_diagram(y_test: np.ndarray,
                 predictions: np.ndarray,
                 label_names: list,
                 save_dir: str,
                 fe_h=False,
                 teff_max=20000,
                 feh_min=-3,
                 name_append=''):
    """
    make a kiel diagram for test and predicted
    """
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))

    binsx = np.linspace(np.nanmin(y_test[:, label_names.index('teff')]), teff_max, 100)
    binsy = np.linspace(0, 5.5, 100)

    if fe_h:
        H_weights, xedges, yedges = np.histogram2d(y_test[:, label_names.index('teff')],
                                                y_test[:, label_names.index('logg')],
                                                bins=[binsx, binsy],
                                                weights=y_test[:, label_names.index('m_h')])

        H_counts, _, _ = np.histogram2d(y_test[:, label_names.index('teff')],
                                        y_test[:, label_names.index('logg')],
                                        bins=(xedges, yedges))

        weighted_average = H_weights / H_counts


        res = ax1.imshow(weighted_average.T, origin='lower', aspect='auto',
                         extent=(binsx.min(), binsx.max(), binsy.min(), binsy.max()), cmap='inferno',
                        vmin=feh_min, vmax=0.3)
        plt.colorbar(res, label='[Fe/H]', ax=ax1)
    else:
        res = ax1.hist2d(y_test[:, label_names.index('teff')], y_test[:, label_names.index('logg')],
                        bins=[binsx, binsy], norm=LogNorm(), cmap='inferno')
        plt.colorbar(res[-1], label='N', ax=ax1)
    ax1.grid()
    ax1.set_title('Test Data')
    ax1.set_xlabel('Teff')
    ax1.set_ylabel('log(g)')
    ax1.invert_xaxis()
    ax1.invert_yaxis()

    if fe_h:
        H_weights, xedges, yedges = np.histogram2d(predictions[:, label_names.index('teff')],
                                                predictions[:, label_names.index('logg')],
                                                bins=[binsx, binsy],
                                                weights=predictions[:, label_names.index('m_h')])

        H_counts, _, _ = np.histogram2d(predictions[:, label_names.index('teff')],
                                        predictions[:, label_names.index('logg')], bins=(xedges, yedges))

        weighted_average = H_weights / H_counts


        res = ax2.imshow(weighted_average.T, origin='lower', aspect='auto',
                         extent=(binsx.min(), binsx.max(), binsy.min(), binsy.max()), cmap='inferno',
                        vmin=feh_min, vmax=0.3)
        plt.colorbar(res, label='[Fe/H]', ax=ax2)
    else:
        res = ax2.hist2d(predictions[:, label_names.index('teff')], predictions[:, label_names.index('logg')],
                        bins=[binsx, binsy], norm=LogNorm(), cmap='inferno')
        plt.colorbar(res[-1], label='N', ax=ax2)
    ax2.grid()
    ax2.set_title('Predictions')
    ax2.set_xlabel('Teff')
    ax2.set_ylabel('log(g)')
    ax2.invert_xaxis()
    ax2.invert_yaxis()
    if fe_h:
        plt.savefig(f'{save_dir}/kiel_diagram_fe_h{name_append}.png')
    else:
        plt.savefig(f'{save_dir}/kiel_diagram{name_append}.png')
    plt.close()


def alpha_fe_plot(y_test: np.ndarray,
                  predictions: np.ndarray,
                  label_names: list,
                  save_dir: str,
                  convert_alpha: bool,
                  feh_min: float = -3.,
                  name_append=''):
    """
    Make plot of alpha/M vs Fe/H for test and predicted
    """
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))

    binsx = np.linspace(feh_min, 0.5, 100)
    binsy = np.linspace(-0.4, 0.5, 100)

    if convert_alpha:
        res = ax1.hist2d(y_test[:, label_names.index('m_h')],
                        y_test[:, label_names.index('alpha_h')] - y_test[:, label_names.index('m_h')],
                        bins=[binsx, binsy], norm=LogNorm(), cmap='inferno')
    else:
        res = ax1.hist2d(y_test[:, label_names.index('m_h')],
                        y_test[:, label_names.index('alpha_m')],
                        bins=[binsx, binsy], norm=LogNorm(), cmap='inferno')
    plt.colorbar(res[-1], label='N', ax=ax1)
    ax1.grid()
    ax1.set_title('Test Data')
    ax1.set_xlabel('Fe/H')
    ax1.set_ylabel('alpha/M')

    if convert_alpha:
        res = ax2.hist2d(predictions[:, label_names.index('m_h')],
                        predictions[:, label_names.index('alpha_h')] - predictions[:, label_names.index('m_h')],
                        bins=[binsx, binsy], norm=LogNorm(), cmap='inferno')
    else:
        res = ax2.hist2d(predictions[:, label_names.index('m_h')],
                        predictions[:, label_names.index('alpha_m')],
                        bins=[binsx, binsy], norm=LogNorm(), cmap='inferno')
    plt.colorbar(res[-1], label='N', ax=ax2)
    ax2.grid()
    ax2.set_title('Predictions')
    ax2.set_xlabel('Fe/H')
    ax2.set_ylabel('alpha/M')
    plt.savefig(f'{save_dir}/alpha_m_vs_fe_h{name_append}.png')
    plt.close()
