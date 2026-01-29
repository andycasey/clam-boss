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
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap
import optax
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm
from sklearn.decomposition import NMF

jax.config.update("jax_enable_x64", True)


def load_data(file_path, convert_alpha=True):
    """Load and preprocess the training data."""
    data = np.load(file_path)['lux_data']

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

    if convert_alpha:
        # Extract stellar labels: teff, logg, m_h, alpha_h
        labels = np.column_stack([
            data['teff'],
            data['logg'],
            data['fe_h'],
            data['raw_alpha_m_atm'] + data['fe_h']  # [alpha/H] = [alpha/M] + [M/H]
        ])
    else:
        # Extract stellar labels: teff, logg, m_h, alpha_h
        labels = np.column_stack([
            data['teff'],
            data['logg'],
            data['fe_h'],
            data['raw_alpha_m_atm']
        ])

    return absorption, norm_flux, norm_ivar, labels


def initialize_nmf(absorption, K, seed=42, max_iter=200):
    """Initialize NMF using sklearn for a good starting point."""
    model = NMF(n_components=K, init='nndsvda', max_iter=max_iter, random_state=seed)
    W_init = model.fit_transform(absorption)
    H_init = model.components_
    return W_init, H_init


def build_design_matrix_jax(labels_std):
    """
    Build design matrix for a single star in JAX.

    For 4 labels, creates 15 features:
    - 1 bias term
    - 4 linear terms
    - 4 quadratic terms
    - 6 cross-terms
    """
    n_labels = labels_std.shape[0]

    features = [jnp.array([1.0])]  # bias

    # Linear terms
    for i in range(n_labels):
        features.append(jnp.array([labels_std[i]]))

    # Quadratic terms
    for i in range(n_labels):
        features.append(jnp.array([labels_std[i] ** 2]))

    # Cross-terms
    for i in range(n_labels):
        for j in range(i + 1, n_labels):
            features.append(jnp.array([labels_std[i] * labels_std[j]]))

    return jnp.concatenate(features)


def build_design_matrix_batch_jax(labels_std_batch):
    """Build design matrix for a batch of stars."""
    return vmap(build_design_matrix_jax)(labels_std_batch)


def build_design_matrix_np(labels_std):
    """Build design matrix in numpy for a batch of stars."""
    n_samples, n_labels = labels_std.shape

    features = [np.ones(n_samples)]  # bias

    # Linear terms
    for i in range(n_labels):
        features.append(labels_std[:, i])

    # Quadratic terms
    for i in range(n_labels):
        features.append(labels_std[:, i] ** 2)

    # Cross-terms
    for i in range(n_labels):
        for j in range(i + 1, n_labels):
            features.append(labels_std[:, i] * labels_std[:, j])

    return np.column_stack(features)


def joint_optimization(flux, ivar, init_labels, K, n_iter=5000, learning_rate=0.01,
                       print_every=500, seed=42, label_weight=1.0):
    """
    Jointly optimize stellar labels, polynomial coefficients, and NMF basis.

    Parameters:
    -----------
    flux : array (n_stars, n_wavelengths)
        Normalized flux spectra
    ivar : array (n_stars, n_wavelengths)
        Inverse variance weights
    init_labels : array (n_stars, 4)
        Initial stellar labels (teff, logg, m_h, alpha_h)
    K : int
        Number of NMF components
    n_iter : int
        Number of optimization iterations
    learning_rate : float
        Learning rate for Adam optimizer
    label_weight : float
        Weight for label loss term (penalizes deviation from initial labels)

    Returns:
    --------
    labels : optimized stellar labels
    theta : polynomial coefficients (15 x K)
    H : NMF basis vectors (K x n_wavelengths)
    """
    n_stars, n_wavelengths = flux.shape
    n_labels = init_labels.shape[1]
    n_features = 1 + n_labels + n_labels + n_labels * (n_labels - 1) // 2  # 15

    print(f"Joint optimization setup:")
    print(f"  Stars: {n_stars}, Wavelengths: {n_wavelengths}")
    print(f"  K = {K} components")
    print(f"  Design matrix features: {n_features}")
    print(f"  Total parameters: {n_stars * n_labels + n_features * K + K * n_wavelengths:,}")

    # Standardize labels
    label_mean = np.mean(init_labels, axis=0)
    label_std = np.std(init_labels, axis=0)
    init_labels_std = (init_labels - label_mean) / label_std

    # Initialize NMF from absorption spectra
    absorption = np.clip(1.0 - flux, 0.0, np.inf)
    W_init, H_init = initialize_nmf(absorption, K, seed=seed, max_iter=200)
    #W_init = np.random.uniform(0, 1, size=W_init.shape)
    #H_init = np.random.uniform(0, 1, size=H_init.shape)

    # Initialize theta from initial W and labels
    design_matrix = build_design_matrix_np(init_labels_std)
    theta_init, _, _, _ = np.linalg.lstsq(design_matrix, W_init, rcond=None)


    # Convert to JAX arrays
    flux_jnp = jnp.array(flux)
    var_jnp = 1.0/jnp.maximum(ivar, 1e-16)
    label_mean_jnp = jnp.array(label_mean)
    label_std_jnp = jnp.array(label_std)
    init_labels_std_jnp = jnp.array(init_labels_std)

    # Parameters to optimize (all unconstrained, we'll apply constraints in forward pass)
    # Use log-space for H to ensure positivity
    params = {
        'labels_std': jnp.array(init_labels_std),
        'theta': jnp.array(theta_init),
        'log_H': jnp.log(jnp.array(H_init) + 1e-10),
        'ln_scatter': 0.1 * jnp.ones(n_wavelengths) # initialize scatter params
    }

    @jit
    def forward(params):
        """Compute predicted flux from parameters."""
        labels_std = params['labels_std']
        theta = params['theta']
        H = jnp.exp(params['log_H'])  # Ensure H >= 0

        # Build design matrix for all stars
        design_matrix = build_design_matrix_batch_jax(labels_std)

        # Predict weights (enforce non-negativity)
        W = jnp.maximum(design_matrix @ theta, 0)

        # Predict flux
        pred_flux = 1.0 - W @ H

        return pred_flux, W, H

    @jit
    def loss_fn(params):
        """Compute weighted reconstruction loss plus label loss."""
        pred_flux, W, H = forward(params)

        # Total variance = data variance + model scatter^2
        scatter_sq = jnp.exp(params['ln_scatter'])**2
        total_var = var_jnp + scatter_sq

        # Negative log-likelihood (Gaussian): 0.5 * [chi^2 + log(var)]
        # The log(var) term penalizes large scatter and prevents trivial solutions
        chi_sq = (flux_jnp - pred_flux)**2 / total_var
        log_term = jnp.log(total_var)
        recon_loss = 0.5 * jnp.sum(chi_sq + log_term) / (n_stars * n_wavelengths)

        # Label loss: penalize deviation from initial labels (normalized by number of labels)
        label_residual = params['labels_std'] - init_labels_std_jnp
        label_loss = jnp.sum(label_residual ** 2) / (n_stars * n_labels)

        return recon_loss + label_weight * label_loss

    @jit
    def loss_and_grad(params):
        return jax.value_and_grad(loss_fn)(params)

    # Optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    @jit
    def update_step(params, opt_state):
        loss, grads = loss_and_grad(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    # Initial loss
    initial_loss = float(loss_fn(params))
    print(f"  Initial loss: {initial_loss:.6e}")

    # Optimization loop
    losses = []
    print(f"\nOptimizing for {n_iter} iterations...")

    with tqdm(total=n_iter) as pb:

        for i in range(n_iter):
            params, opt_state, loss = update_step(params, opt_state)

            #if (i + 1) % print_every == 0:
            losses.append(float(loss))
            pb.set_description(f"loss = {float(loss):.4e}")
            pb.update()

    final_loss = float(loss_fn(params))
    print(f"  Final loss: {final_loss:.6e}")

    # Extract final parameters
    labels_std_final = np.array(params['labels_std'])
    labels_final = labels_std_final * label_std + label_mean
    theta_final = np.array(params['theta'])
    H_final = np.exp(np.array(params['log_H']))
    scatter = np.exp(np.array(params['ln_scatter']))

    # Compute final W
    design_matrix_final = build_design_matrix_np(labels_std_final)
    W_final = np.maximum(design_matrix_final @ theta_final, 0)

    return labels_final, theta_final, H_final, W_final, label_mean, label_std, losses, scatter


def infer_labels(flux, ivar, theta, H, label_mean, label_std, scatter, init_labels_std=None,
                 n_iter=2000, learning_rate=0.05, seed=42, optimizer='adam',
                 grid_points=5, grid_range=(-3.0, 3.0)):
    """
    Infer stellar labels from spectra using a trained model.

    Given a trained model (theta, H, scatter) and new spectra (flux, ivar),
    optimize only the stellar labels while keeping the model fixed.

    Parameters:
    -----------
    flux : array (n_stars, n_wavelengths)
        Normalized flux spectra
    ivar : array (n_stars, n_wavelengths)
        Inverse variance weights
    theta : array (15, K)
        Polynomial coefficients (fixed)
    H : array (K, n_wavelengths)
        NMF basis vectors (fixed)
    label_mean : array (4,)
        Label means for standardization
    label_std : array (4,)
        Label stds for standardization
    scatter : array (n_wavelengths,)
        Model scatter per wavelength
    init_labels_std : array (n_stars, 4) or None
        Initial standardized labels. If None, performs grid search to find
        best starting point for each spectrum.
    n_iter : int
        Number of optimization iterations (for Adam) or max iterations (for BFGS)
    learning_rate : float
        Learning rate for Adam optimizer (ignored for BFGS)
    seed : int
        Random seed
    optimizer : str
        Optimization method: 'adam' or 'bfgs'
    grid_points : int
        Number of grid points per dimension for initial grid search
        (only used when init_labels_std is None)
    grid_range : tuple
        (min, max) range in standardized coordinates for grid search

    Returns:
    --------
    labels : array (n_stars, 4)
        Inferred stellar labels (teff, logg, m_h, alpha_h)
    """
    from scipy.optimize import minimize
    from itertools import product

    n_stars, n_wavelengths = flux.shape
    n_labels = len(label_mean)

    print(f"Inferring labels for {n_stars} stars using {optimizer.upper()}...")

    # Convert to JAX arrays
    flux_jnp = jnp.array(flux)
    var_jnp = 1.0 / jnp.maximum(ivar, 1e-16)
    theta_jnp = jnp.array(theta)
    H_jnp = jnp.array(H)
    scatter_sq = jnp.array(scatter)**2

    # Single-star loss function for grid search and BFGS
    @jit
    def single_star_loss(labels_std_single, flux_single, var_single):
        """Compute loss for a single star."""
        design_vec = build_design_matrix_jax(labels_std_single)
        W = jnp.maximum(design_vec @ theta_jnp, 0)
        pred_flux = 1.0 - W @ H_jnp
        total_var = var_single + scatter_sq
        chi_sq = (flux_single - pred_flux)**2 / total_var
        return 0.5 * jnp.sum(chi_sq)

    single_star_loss_and_grad = jit(jax.value_and_grad(single_star_loss))

    # Grid search to find initial values if not provided
    if init_labels_std is None:
        print(f"  Performing grid search ({grid_points}^{n_labels} = {grid_points**n_labels} points per star)...")

        # Build grid
        grid_1d = np.linspace(grid_range[0], grid_range[1], grid_points)
        grid_points_all = np.array(list(product(*[grid_1d]*n_labels)))  # (n_grid, n_labels)
        n_grid = len(grid_points_all)

        init_labels_std = np.zeros((n_stars, n_labels))

        for i in tqdm(range(n_stars), desc="Grid search"):
            flux_i = jnp.array(flux[i])
            var_i = jnp.array(1.0 / np.maximum(ivar[i], 1e-16))

            best_loss = np.inf
            best_point = grid_points_all[0]

            for grid_point in grid_points_all:
                loss_val = float(single_star_loss(jnp.array(grid_point), flux_i, var_i))
                if loss_val < best_loss:
                    best_loss = loss_val
                    best_point = grid_point

            init_labels_std[i] = best_point

        print(f"  Grid search complete.")

    if optimizer == 'bfgs':
        # BFGS optimization for each star independently
        print(f"  Running L-BFGS-B optimization...")
        labels_std_final = np.zeros((n_stars, n_labels))

        for i in tqdm(range(n_stars), desc="BFGS"):
            flux_i = jnp.array(flux[i])
            var_i = jnp.array(1.0 / np.maximum(ivar[i], 1e-16))
            x0 = init_labels_std[i]

            def objective(x):
                loss, grad = single_star_loss_and_grad(jnp.array(x), flux_i, var_i)
                return float(loss), np.array(grad)

            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': n_iter, 'disp': False}
            )
            labels_std_final[i] = result.x

    else:  # adam
        # Batch Adam optimization (original behavior)
        params = {'labels_std': jnp.array(init_labels_std)}

        @jit
        def forward(params):
            """Compute predicted flux from labels."""
            labels_std = params['labels_std']
            design_matrix = build_design_matrix_batch_jax(labels_std)
            W = jnp.maximum(design_matrix @ theta_jnp, 0)
            pred_flux = 1.0 - W @ H_jnp
            return pred_flux

        @jit
        def loss_fn(params):
            """Compute weighted reconstruction loss."""
            pred_flux = forward(params)
            total_var = var_jnp + scatter_sq
            chi_sq = (flux_jnp - pred_flux)**2 / total_var
            return 0.5 * jnp.sum(chi_sq) / (n_stars * n_wavelengths)

        @jit
        def loss_and_grad(params):
            return jax.value_and_grad(loss_fn)(params)

        opt = optax.adam(learning_rate)
        opt_state = opt.init(params)

        @jit
        def update_step(params, opt_state):
            loss, grads = loss_and_grad(params)
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        with tqdm(total=n_iter) as pb:
            for i in range(n_iter):
                params, opt_state, loss = update_step(params, opt_state)
                pb.set_description(f"loss = {float(loss):.4e}")
                pb.update()

        labels_std_final = np.array(params['labels_std'])

    # Convert back to physical labels
    labels_final = labels_std_final * label_std + label_mean

    return labels_final


def plot_test_comparison(true_labels, inferred_labels, label_names, save_path):
    """Create comparison plots of true vs inferred labels for test set."""
    n_labels = true_labels.shape[1]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()

    label_bounds = {
        'teff': (3500, 7500),
        'logg': (0.5, 5.5),
        'm_h': (-2.5, 0.75),
        'alpha_h': (-0.5, 0.6)
    }

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
    print(f"Saved test comparison plot to {save_path}")


def compute_label_statistics(true_labels, inferred_labels, label_names):
    """Compute bias, scatter, and MAD for each label."""
    stats = {}
    for i, name in enumerate(label_names):
        diff = inferred_labels[:, i] - true_labels[:, i]
        valid = np.isfinite(diff)
        bias = np.median(diff[valid])
        scatter = np.std(diff[valid])
        mad = np.median(np.abs(diff[valid] - bias))
        stats[name] = {'bias': bias, 'scatter': scatter, 'mad': mad}
    return stats


def plot_comparison(true_labels, inferred_labels, label_names, save_path):
    """Create comparison plots of true vs inferred labels."""
    n_labels = true_labels.shape[1]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()

    label_bounds = {
        'teff': (3500, 7500),
        'logg': (0.5, 5.5),
        'm_h': (-2.5, 0.75),
        'alpha_h': (-0.5, 0.6)
    }

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
    print(f"Saved comparison plot to {save_path}")


def plot_nmf_components(H, save_path):
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
    print(f"Saved NMF components plot to {save_path}")


def plot_loss(losses, save_path):
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
    print(f"Saved loss plot to {save_path}")

def plot_model_scatter(wavelength, scatter, save_path):
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
    print(f"Saved model scatter plot to {save_path}")



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
        inf_weights = np.maximum(design_vec @ theta, 0)[0]
        model_flux = 1.0 - inf_weights @ H

        # Model flux from true labels
        true_labels_std = (true_labels[i] - label_mean) / label_std
        design_vec_true = build_design_matrix_np(true_labels_std.reshape(1, -1))
        true_weights = np.maximum(design_vec_true @ theta, 0)[0]
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
    print(f"Saved spectra comparison to {save_path}")


def plot_residual_histograms(true_labels, inferred_labels, label_names, save_path):
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
    print(f"Saved residual histogram to {save_path}")


if __name__ == '__main__':
    # Configuration
    data_file = 'boss_apogee_lux_training_data.npz'
    K = 32
    n_iter = 10_000
    learning_rate = 0.01 # 0.1 is too aggressive
    print_every = 1000
    convert_alpha = False  # if True, convert to alpha/h

    if convert_alpha:
        label_names = ['teff', 'logg', 'm_h', 'alpha_h']
        output_dir = 'nmf_joint_results_with_scatter_K32'
    else:
        label_names = ['teff', 'logg', 'm_h', 'alpha_m']
        output_dir = 'nmf_joint_results_with_scatter_K32_alpha_m'

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving results to: {output_dir}/")

    print("=" * 60)
    print("Joint NMF Stellar Spectra Model")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  K = {K} components")
    print(f"  Iterations = {n_iter}")
    print(f"  Learning rate = {learning_rate}")

    # Load data
    print("\n[1/4] Loading data...")
    absorption, flux, ivar, true_labels = load_data(data_file,
                                                    convert_alpha=convert_alpha)
    n_stars, n_wavelengths = flux.shape
    print(f"  Loaded {n_stars} stars with {n_wavelengths} wavelength pixels")

    # Joint optimization
    print("\n[2/4] Running joint optimization...")
    inferred_labels, theta, H, W, label_mean, label_std, losses, scatter = joint_optimization(
        flux, ivar, true_labels, K,
        n_iter=n_iter,
        learning_rate=learning_rate,
        print_every=print_every,
        seed=42
    )

    # Compute statistics
    print("\n[3/4] Computing statistics...")
    stats = compute_label_statistics(true_labels, inferred_labels, label_names)

    print("\n" + "=" * 60)
    print("Summary Statistics (Training Set)")
    print("=" * 60)
    for name in label_names:
        s = stats[name]
        print(f"  {name:8s}: bias={s['bias']:+.4f}, scatter={s['scatter']:.4f}, MAD={s['mad']:.4f}")

    # Generate plots
    print("\n[4/4] Generating plots...")

    # Wavelength grid
    loglam = 3.5523 + 0.0001 * np.arange(n_wavelengths)
    wavelength = 10**loglam

    plot_comparison(true_labels, inferred_labels, label_names, f'{output_dir}/label_comparison.png')
    plot_residual_histograms(true_labels, inferred_labels, label_names, f'{output_dir}/label_residuals.png')
    plot_nmf_components(H, f'{output_dir}/nmf_components.png')
    if len(losses) > 1:
        plot_loss(losses, f'{output_dir}/optimization_loss.png')
    plot_spectra_comparison(
        flux, ivar, true_labels, inferred_labels,
        theta, H, label_mean, label_std, wavelength,
        f'{output_dir}/spectra_comparison.png', n_plot=20
    )
    plot_model_scatter(
        wavelength, scatter,
        f'{output_dir}/model_scatter.png'
    )

    # Save results
    print("\nSaving results...")
    np.savez(f'{output_dir}/joint_model_results.npz',
             inferred_labels=inferred_labels,
             true_labels=true_labels,
             theta=theta,
             H=H,
             W=W,
             label_mean=label_mean,
             label_std=label_std,
             losses=losses,
             scatter=scatter,
             stats=stats)

    print(f"  Saved to {output_dir}/")

    # =========================================================================
    # TEST STEP: Infer labels from spectra alone (no known labels)
    # =========================================================================
    print("\n" + "=" * 60)
    print("Test Step: Inferring labels from spectra (no known labels)")
    print("=" * 60)

    # Use all data as "test" - pretend we don't know the labels
    # In practice you'd use a held-out test set
    test_flux = flux
    test_ivar = ivar
    test_true_labels = true_labels

    print(f"Test set: {len(test_flux)} spectra")
    print("Inferring labels using trained model (theta, H fixed)...")

    # Infer labels using only flux and ivar
    test_inferred_labels = infer_labels(
        test_flux, test_ivar,
        theta, H, label_mean, label_std, scatter,
        init_labels_std=(test_true_labels - label_mean) / label_std,
        n_iter=3000,
        learning_rate=0.05,
        optimizer='adam'
    )

    # Compute test statistics
    test_stats = compute_label_statistics(test_true_labels, test_inferred_labels, label_names)

    print("\n" + "=" * 60)
    print("Test Set Statistics")
    print("=" * 60)
    for name in label_names:
        s = test_stats[name]
        print(f"  {name:8s}: bias={s['bias']:+.4f}, scatter={s['scatter']:.4f}, MAD={s['mad']:.4f}")

    # Plot test results: true vs inferred
    print("\nGenerating test comparison plot...")
    plot_test_comparison(
        test_true_labels, test_inferred_labels, label_names,
        f'{output_dir}/test_true_vs_inferred.png'
    )

    # Save test results
    np.savez(f'{output_dir}/test_inference_results.npz',
             test_true_labels=test_true_labels,
             test_inferred_labels=test_inferred_labels,
             test_stats=test_stats)
    print(f"Saved test results to {output_dir}/test_inference_results.npz")

    print("\nDone!")
