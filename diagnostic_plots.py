"""
Diagnostic plots for checking quality of inferred stellar parameters.

Creates corner-style chi-squared maps around the inferred solution,
plus spectrum comparison plot.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from itertools import combinations
from nmf_stellar_model_joint import build_design_matrix_np


def compute_chi_sq(flux, ivar, labels_std, theta, H, scatter):
    """
    Compute chi-squared for a single spectrum given standardized labels.
    
    Parameters
    ----------
    flux : array (n_wavelengths,)
        Observed normalized flux
    ivar : array (n_wavelengths,)
        Inverse variance
    labels_std : array (n_labels,)
        Standardized stellar labels
    theta : array (n_features, K)
        Polynomial coefficients
    H : array (K, n_wavelengths)
        NMF basis vectors
    scatter : array (n_wavelengths,)
        Model scatter per wavelength
        
    Returns
    -------
    chi_sq : float
        Chi-squared value
    """
    design_vec = build_design_matrix_np(labels_std.reshape(1, -1))
    W = np.maximum(design_vec @ theta, 0)[0]
    pred_flux = 1.0 - W @ H
    
    var = 1.0 / np.maximum(ivar, 1e-16)
    total_var = var + scatter**2
    
    chi_sq = np.sum((flux - pred_flux)**2 / total_var)
    return chi_sq


def compute_chi_sq_grid_2d(flux, ivar, inferred_labels_std, theta, H, scatter,
                           param_i, param_j, grid_i, grid_j):
    """
    Compute chi-squared on a 2D grid for parameters i and j,
    minimizing over all other parameters.
    
    For simplicity, we fix the other parameters at their inferred values
    (true marginalization would require a full grid search, which is expensive).
    
    Parameters
    ----------
    flux : array (n_wavelengths,)
    ivar : array (n_wavelengths,)
    inferred_labels_std : array (n_labels,)
        Inferred standardized labels (best-fit position)
    theta, H, scatter : model parameters
    param_i, param_j : int
        Indices of the two parameters to vary
    grid_i, grid_j : array
        Grid values for parameters i and j (in standardized units)
        
    Returns
    -------
    chi_sq_grid : array (len(grid_i), len(grid_j))
        Chi-squared values on the 2D grid
    """
    n_labels = len(inferred_labels_std)
    chi_sq_grid = np.zeros((len(grid_i), len(grid_j)))
    
    for ii, val_i in enumerate(grid_i):
        for jj, val_j in enumerate(grid_j):
            # Start from inferred values
            test_labels = inferred_labels_std.copy()
            test_labels[param_i] = val_i
            test_labels[param_j] = val_j
            
            chi_sq_grid[ii, jj] = compute_chi_sq(
                flux, ivar, test_labels, theta, H, scatter
            )
    
    return chi_sq_grid


def plot_chi_sq_diagnostic(flux, ivar, inferred_labels, theta, H, 
                           label_mean, label_std, scatter, wavelength=None,
                           n_grid=31, delta_std=2.0, save_path=None,
                           label_names=None):
    """
    Create diagnostic corner plot showing chi-squared maps around inferred solution.
    
    Parameters
    ----------
    flux : array (n_wavelengths,)
        Observed normalized flux
    ivar : array (n_wavelengths,)
        Inverse variance
    inferred_labels : array (n_labels,)
        Inferred stellar labels in physical units
    theta : array (n_features, K)
        Polynomial coefficients
    H : array (K, n_wavelengths)
        NMF basis vectors
    label_mean, label_std : arrays (n_labels,)
        Standardization parameters
    scatter : array (n_wavelengths,)
        Model scatter per wavelength
    wavelength : array (n_wavelengths,), optional
        Wavelength array for spectrum plot
    n_grid : int
        Number of grid points per dimension
    delta_std : float
        Half-width of grid in standardized units (default: ±2 sigma)
    save_path : str, optional
        Path to save the figure
    label_names : list of str, optional
        Names for each parameter
        
    Returns
    -------
    fig : matplotlib Figure
    """
    n_labels = len(inferred_labels)
    if label_names is None:
        label_names = ['teff', 'logg', '[M/H]', '[α/H]']
    
    # Convert to standardized units
    inferred_labels_std = (inferred_labels - label_mean) / label_std
    
    # Compute best-fit model spectrum
    design_vec = build_design_matrix_np(inferred_labels_std.reshape(1, -1))
    W_best = np.maximum(design_vec @ theta, 0)[0]
    model_flux = 1.0 - W_best @ H
    
    # Compute chi-squared at best fit
    chi_sq_best = compute_chi_sq(flux, ivar, inferred_labels_std, theta, H, scatter)
    
    # Set up figure with GridSpec
    # Corner plot is (n_labels-1) x (n_labels-1), plus bottom row for spectrum
    fig = plt.figure(figsize=(14, 16))
    
    # Create grid: n_labels rows, n_labels-1 columns
    # Bottom row spans all columns for spectrum plot
    gs = GridSpec(n_labels, n_labels - 1, figure=fig, 
                  height_ratios=[1]*(n_labels-1) + [1.2],
                  hspace=0.3, wspace=0.3)
    
    # Parameter pairs for corner plot (lower triangle)
    # Row i shows parameter i+1, columns show parameters 0 to i
    pair_axes = {}
    
    for row in range(n_labels - 1):
        param_y = row + 1  # y-axis parameter index
        for col in range(row + 1):
            param_x = col  # x-axis parameter index
            
            ax = fig.add_subplot(gs[row, col])
            pair_axes[(param_x, param_y)] = ax
            
            # Create grids around inferred values
            grid_x = np.linspace(
                inferred_labels_std[param_x] - delta_std,
                inferred_labels_std[param_x] + delta_std,
                n_grid
            )
            grid_y = np.linspace(
                inferred_labels_std[param_y] - delta_std,
                inferred_labels_std[param_y] + delta_std,
                n_grid
            )
            
            # Compute chi-squared grid
            chi_sq_grid = compute_chi_sq_grid_2d(
                flux, ivar, inferred_labels_std, theta, H, scatter,
                param_x, param_y, grid_x, grid_y
            )
            
            # Convert grid to physical units for plotting
            grid_x_phys = grid_x * label_std[param_x] + label_mean[param_x]
            grid_y_phys = grid_y * label_std[param_y] + label_mean[param_y]
            
            # Plot chi-squared map (relative to minimum)
            chi_sq_rel = chi_sq_grid - chi_sq_grid.min()
            
            # Use imshow with extent
            extent = [grid_x_phys[0], grid_x_phys[-1], 
                      grid_y_phys[0], grid_y_phys[-1]]
            
            im = ax.imshow(chi_sq_rel.T, origin='lower', extent=extent,
                          aspect='auto', cmap='viridis_r')
            
            # Add contours at delta chi-sq = 1, 4, 9 (1, 2, 3 sigma for 1 dof)
            # For 2 dof: 2.30, 6.17, 11.8 (68%, 95%, 99.7%)
            contour_levels = [2.30, 6.17, 11.8]
            try:
                ax.contour(grid_x_phys, grid_y_phys, chi_sq_rel.T, 
                          levels=contour_levels, colors='white', 
                          linewidths=0.8, linestyles=['solid', 'dashed', 'dotted'])
            except:
                pass  # Contours may fail if chi-sq surface is flat
            
            # Mark inferred position
            ax.axvline(inferred_labels[param_x], color='red', lw=1, ls='--', alpha=0.7)
            ax.axhline(inferred_labels[param_y], color='red', lw=1, ls='--', alpha=0.7)
            ax.plot(inferred_labels[param_x], inferred_labels[param_y], 
                   'r+', ms=10, mew=2)
            
            # Labels
            if row == n_labels - 2:  # Bottom row of corner plot
                ax.set_xlabel(label_names[param_x], fontsize=10)
            else:
                ax.set_xticklabels([])
                
            if col == 0:  # Left column
                ax.set_ylabel(label_names[param_y], fontsize=10)
            else:
                ax.set_yticklabels([])
    
    # Add colorbar to the right of corner plot
    cbar_ax = fig.add_axes([0.92, 0.45, 0.02, 0.35])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Δχ²', fontsize=10)
    
    # Spectrum plot at bottom spanning all columns
    ax_spec = fig.add_subplot(gs[n_labels - 1, :])
    
    if wavelength is None:
        # Default BOSS wavelength grid
        n_wavelengths = len(flux)
        loglam = 3.5523 + 0.0001 * np.arange(n_wavelengths)
        wavelength = 10**loglam
    
    # Plot observed spectrum
    good = ivar > 0
    ax_spec.plot(wavelength[good], flux[good], 'k-', lw=0.5, alpha=0.7, 
                label='Observed')
    
    # Plot model spectrum
    ax_spec.plot(wavelength, model_flux, 'r-', lw=0.8, alpha=0.8, 
                label='Model')
    
    # Plot residuals (offset)
    residuals = flux - model_flux
    offset = -0.3
    ax_spec.plot(wavelength[good], residuals[good] + offset, 'b-', lw=0.3, 
                alpha=0.5, label=f'Residuals (offset {offset})')
    ax_spec.axhline(offset, color='gray', lw=0.5, ls='--')
    
    ax_spec.set_xlabel('Wavelength (Å)', fontsize=11)
    ax_spec.set_ylabel('Normalized Flux', fontsize=11)
    ax_spec.set_ylim(-0.5, 1.15)
    ax_spec.legend(loc='upper right', fontsize=9)
    ax_spec.grid(True, alpha=0.3)
    
    # Title with inferred parameters and chi-squared
    title = (f"Inferred: Teff={inferred_labels[0]:.0f} K, "
             f"log g={inferred_labels[1]:.2f}, "
             f"[M/H]={inferred_labels[2]:.2f}, "
             f"[α/H]={inferred_labels[3]:.2f}  |  "
             f"χ²={chi_sq_best:.1f}")
    ax_spec.set_title(title, fontsize=11)
    
    plt.suptitle('Chi-squared Diagnostic Plot', fontsize=14, y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved diagnostic plot to {save_path}")
    
    return fig


def plot_chi_sq_diagnostic_batch(flux_batch, ivar_batch, inferred_labels_batch,
                                 theta, H, label_mean, label_std, scatter,
                                 output_dir, indices=None, wavelength=None,
                                 n_grid=21, delta_std=2.0, label_names=None):
    """
    Create diagnostic plots for multiple spectra.
    
    Parameters
    ----------
    flux_batch : array (n_stars, n_wavelengths)
    ivar_batch : array (n_stars, n_wavelengths)
    inferred_labels_batch : array (n_stars, n_labels)
    theta, H, label_mean, label_std, scatter : model parameters
    output_dir : str
        Directory to save plots
    indices : list of int, optional
        Which spectra to plot. If None, plots first 10.
    wavelength : array, optional
    n_grid : int
    delta_std : float
    label_names : list of str, optional
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    if indices is None:
        indices = list(range(min(10, len(flux_batch))))
    
    for i in indices:
        save_path = os.path.join(output_dir, f'diagnostic_star_{i:04d}.png')
        plot_chi_sq_diagnostic(
            flux_batch[i], ivar_batch[i], inferred_labels_batch[i],
            theta, H, label_mean, label_std, scatter,
            wavelength=wavelength, n_grid=n_grid, delta_std=delta_std,
            save_path=save_path, label_names=label_names
        )
        plt.close('all')
    
    print(f"Saved {len(indices)} diagnostic plots to {output_dir}/")


if __name__ == '__main__':
    # Example usage with saved model results
    import os
    
    # Load model
    model_dir = 'nmf_joint_results_with_scatter_K32'
    if not os.path.exists(f'{model_dir}/joint_model_results.npz'):
        print(f"Model not found in {model_dir}/")
        print("Run nmf_stellar_model_joint.py first to train the model.")
        exit(1)
    
    res = np.load(f'{model_dir}/joint_model_results.npz', allow_pickle=True)
    theta = res['theta']
    H = res['H']
    label_mean = res['label_mean']
    label_std = res['label_std']
    scatter = res['scatter']
    
    # Load some test data
    from nmf_stellar_model_joint import load_data
    data_file = 'boss_apogee_lux_training_data.npz'
    if not os.path.exists(data_file):
        print(f"Data file {data_file} not found.")
        exit(1)
    
    absorption, flux, ivar, true_labels = load_data(data_file)
    
    # Get inferred labels (for demo, use true labels or load from results)
    inferred_labels = res['inferred_labels']
    
    # Wavelength grid
    n_wavelengths = flux.shape[1]
    loglam = 3.5523 + 0.0001 * np.arange(n_wavelengths)
    wavelength = 10**loglam
    
    # Create diagnostic plots for a few random stars
    np.random.seed(42)
    indices = np.random.choice(len(flux), size=5, replace=False)
    
    output_dir = 'diagnostic_plots'
    plot_chi_sq_diagnostic_batch(
        flux, ivar, inferred_labels,
        theta, H, label_mean, label_std, scatter,
        output_dir=output_dir,
        indices=indices,
        wavelength=wavelength,
        n_grid=25,
        delta_std=2.5
    )
