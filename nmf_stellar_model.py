"""
NMF-based stellar spectra model.

This script:
1. Loads BOSS/APOGEE training data
2. Computes absorption spectra from flux
3. Performs Non-negative Matrix Factorization (NMF) with K components
4. Trains a model from stellar labels to NMF weights
5. Builds a forward model to predict spectra from stellar labels
6. Infers stellar parameters for all training stars
7. Generates comparison plots
"""

import os
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, vmap
import optax
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import NMF

jax.config.update("jax_enable_x64", True)


def load_data(file_path):
    """Load and preprocess the training data."""
    data = np.load(file_path)['lux_data']

    # Normalize flux
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

    # Extract stellar labels: teff, logg, m_h (fe_h), alpha_h
    labels = np.column_stack([
        data['teff'],
        data['logg'],
        data['fe_h'],
        data['raw_alpha_m_atm'] + data['fe_h']  # [alpha/H] = [alpha/M] + [M/H]
    ])

    return absorption, norm_flux, norm_ivar, labels


def weighted_nmf(X, ivar, K, n_iter=10000, seed=42, tol=1e-8,
                 l1_alpha=0.01, ortho_alpha=0.001):
    """
    Perform weighted NMF using multiplicative update rules with regularization.

    Minimizes: sum_ij ivar_ij * (X_ij - (WH)_ij)^2
               + l1_alpha * ||W||_1
               + ortho_alpha * ||H @ H.T - diag||^2

    Returns W, H in the ORIGINAL space (not weighted), so:
        absorption ≈ W @ H
        flux ≈ 1 - W @ H
    """
    print(f"Running weighted NMF with K={K} for up to {n_iter} iterations...")
    print(f"  L1 regularization on W: {l1_alpha}")
    print(f"  Orthogonality regularization on H: {ortho_alpha}")

    n_samples, n_features = X.shape
    eps = 1e-10

    # Initialize using sklearn's NNDSVD (on unweighted data)
    model_init = NMF(n_components=K, init='nndsvda', max_iter=1, random_state=seed)
    W = model_init.fit_transform(X) + eps
    H = model_init.components_ + eps

    # Ensure non-negative weights
    weights = np.maximum(ivar, 0)

    # Precompute weighted X
    wX = weights * X

    losses = []
    for iteration in tqdm(range(n_iter)):
        # Update H with orthogonality regularization
        # Standard update: H <- H * (W.T @ (weights * X)) / (W.T @ (weights * WH) + eps)
        # Orthogonality penalty gradient: 2 * ortho_alpha * (H @ H.T - diag) @ H
        WH = W @ H
        numerator_H = W.T @ wX
        denominator_H = W.T @ (weights * WH) + eps

        if ortho_alpha > 0:
            # Add orthogonality penalty to denominator
            HHT = H @ H.T
            HHT_offdiag = HHT - np.diag(np.diag(HHT))
            ortho_grad = 2 * ortho_alpha * (HHT_offdiag @ H)
            denominator_H = denominator_H + np.maximum(ortho_grad, 0) + eps

        H *= (numerator_H / denominator_H)
        np.maximum(H, eps, out=H)

        # Update W with L1 regularization
        # Standard update: W <- W * ((weights * X) @ H.T) / ((weights * WH) @ H.T + eps)
        # L1 penalty adds constant to denominator
        WH = W @ H
        numerator_W = wX @ H.T
        denominator_W = (weights * WH) @ H.T + l1_alpha + eps
        W *= (numerator_W / denominator_W)
        np.maximum(W, eps, out=W)

        # Compute loss periodically
        if iteration % 100 == 0:
            WH = W @ H
            residual = X - WH
            recon_loss = np.sum(weights * residual**2)
            l1_loss = l1_alpha * np.sum(np.abs(W))
            ortho_loss = ortho_alpha * np.sum(HHT_offdiag**2) if ortho_alpha > 0 else 0
            total_loss = recon_loss + l1_loss + ortho_loss
            losses.append(total_loss)

            if len(losses) > 1:
                rel_change = abs(losses[-2] - losses[-1]) / (abs(losses[-2]) + eps)
                if rel_change < tol:
                    print(f"\n  Converged at iteration {iteration} (rel_change={rel_change:.2e})")
                    break

    # Final loss
    WH = W @ H
    residual = X - WH
    final_loss = np.sum(weights * residual**2)
    print(f"  Final reconstruction loss: {final_loss:.6e}")

    # Report W statistics
    print(f"  W range: [{W.min():.4f}, {W.max():.4f}], mean={W.mean():.4f}, std={W.std():.4f}")
    print(f"  W sparsity (fraction < 0.01): {(W < 0.01).mean():.1%}")

    return W, H, losses


def build_design_matrix(labels_standardized):
    """
    Build design matrix with bias, linear, quadratic, and cross-terms.

    For 4 labels (teff, logg, m_h, alpha_h), this creates:
    - 1 bias term
    - 4 linear terms
    - 4 quadratic terms (x^2)
    - 6 cross-terms (x_i * x_j for i < j)
    Total: 15 features
    """
    n_samples, n_labels = labels_standardized.shape

    # Start with bias
    features = [np.ones(n_samples)]

    # Linear terms
    for i in range(n_labels):
        features.append(labels_standardized[:, i])

    # Quadratic terms
    for i in range(n_labels):
        features.append(labels_standardized[:, i] ** 2)

    # Cross-terms
    for i in range(n_labels):
        for j in range(i + 1, n_labels):
            features.append(labels_standardized[:, i] * labels_standardized[:, j])

    return np.column_stack(features)


def train_label_to_weights_linear(labels, weights):
    """
    Train a polynomial model: weights = design_matrix @ theta

    Includes: bias, linear, quadratic, and cross-terms.
    """
    # Standardize labels
    label_mean = np.mean(labels, axis=0)
    label_std = np.std(labels, axis=0)
    labels_standardized = (labels - label_mean) / label_std

    # Build design matrix with all terms
    design_matrix = build_design_matrix(labels_standardized)
    n_features = design_matrix.shape[1]
    print(f"  Design matrix: {design_matrix.shape[0]} samples x {n_features} features")
    print(f"    (1 bias + 4 linear + 4 quadratic + 6 cross-terms)")

    # Solve least squares
    theta, residuals, rank, s = np.linalg.lstsq(design_matrix, weights, rcond=None)

    # Compute R² score
    predicted = design_matrix @ theta
    ss_res = np.sum((weights - predicted) ** 2)
    ss_tot = np.sum((weights - np.mean(weights, axis=0)) ** 2)
    r2 = 1 - ss_res / ss_tot
    print(f"  Polynomial model R² score: {r2:.4f}")

    return theta, label_mean, label_std


def predict_weights_from_labels_linear(labels, theta, label_mean, label_std):
    """Predict NMF weights from stellar labels using polynomial model."""
    labels_standardized = (labels - label_mean) / label_std
    design_matrix = build_design_matrix(labels_standardized)
    predicted_weights = design_matrix @ theta
    return np.clip(predicted_weights, 0, np.inf)


def diagnose_label_to_weights_fit(W, predicted_W, K):
    """Diagnose how well labels predict NMF weights."""
    print("\n  Component-wise R² (how well labels predict each NMF weight):")
    r2_values = []
    for k in range(K):
        var_total = np.var(W[:, k])
        var_residual = np.var(W[:, k] - predicted_W[:, k])
        r2 = 1 - var_residual / (var_total + 1e-10)
        r2_values.append(r2)
        if k < 10 or k >= K - 3:
            print(f"    Component {k:2d}: R² = {r2:.3f}")
        elif k == 10:
            print(f"    ...")

    total_var = np.var(W)
    residual_var = np.var(W - predicted_W)
    overall_r2 = 1 - residual_var / total_var
    print(f"  Overall variance explained: {100*overall_r2:.1f}%")
    print(f"  Mean component R²: {np.mean(r2_values):.3f}")
    print(f"  Median component R²: {np.median(r2_values):.3f}")

    return r2_values, overall_r2


def infer_labels_batch(flux, ivars, H, predict_weights_fn, label_mean, label_std,
                       n_iter=5000, learning_rate=0.01, init_labels=None):
    """
    Infer stellar labels for a batch of spectra using JAX.

    predict_weights_fn: function that takes (labels_std) and returns weights
                        This allows using either linear or NN model
    """
    n_stars = flux.shape[0]
    H_jnp = jnp.array(H)
    label_mean_jnp = jnp.array(label_mean)
    label_std_jnp = jnp.array(label_std)
    flux_jnp = jnp.array(flux)
    ivars_jnp = jnp.array(ivars)

    # Compute mask (sqrt of ivar)
    masks = jnp.sqrt(jnp.maximum(ivars_jnp, 0))

    def single_star_loss(labels_std, flux, mask):
        """Loss function for single star."""
        weights = predict_weights_fn(labels_std)
        weights = jnp.maximum(weights, 0)
        pred_flux = 1 - weights @ H_jnp
        residual = mask * (flux - pred_flux)
        return jnp.sum(residual**2)

    # Vectorize over stars
    @jit
    def batch_loss(all_labels_std):
        losses = vmap(single_star_loss)(all_labels_std, flux_jnp, masks)
        return jnp.sum(losses)

    # Initialize labels
    if init_labels is not None:
        init_labels_jnp = jnp.array(init_labels)
        all_labels_std = (init_labels_jnp - label_mean_jnp) / label_std_jnp
        print(f"  Using provided initial labels")
    else:
        solar_labels = jnp.array([5500., 4.0, 0.0, 0.0])
        init_labels_std = (solar_labels - label_mean_jnp) / label_std_jnp
        all_labels_std = jnp.tile(init_labels_std, (n_stars, 1))
        print(f"  Using solar-like initial labels")

    # Compute initial loss
    initial_loss = float(batch_loss(all_labels_std))
    print(f"  Initial loss: {initial_loss:.6e}")

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(all_labels_std)

    @jit
    def update_step(all_labels_std, opt_state):
        loss, grads = jax.value_and_grad(batch_loss)(all_labels_std)
        updates, opt_state = optimizer.update(grads, opt_state, all_labels_std)
        all_labels_std = optax.apply_updates(all_labels_std, updates)
        return all_labels_std, opt_state, loss

    print(f"  Inferring labels for {n_stars} stars...")
    for i in tqdm(range(n_iter)):
        all_labels_std, opt_state, loss = update_step(all_labels_std, opt_state)
        if (i + 1) % 1000 == 0:
            print(f"  Iteration {i+1}: total loss = {loss:.6e}")

    print(f"  Final loss: {float(loss):.6e}")

    # Convert back to physical units
    inferred_labels = all_labels_std * label_std_jnp + label_mean_jnp
    return np.array(inferred_labels)


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

        # Compute statistics
        diff = inferred_vals - true_vals
        bias = np.nanmedian(diff)
        scatter = np.nanstd(diff)
        mad = np.nanmedian(np.abs(diff - bias))

        # Plot
        ax.scatter(true_vals, inferred_vals, alpha=0.3, s=2, c='black')

        # 1:1 line
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


def plot_nmf_loss(losses, save_path):
    """Plot NMF training loss."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(losses)
    ax1.set_xlabel('Iteration (x100)')
    ax1.set_ylabel('Total Loss')
    ax1.set_title('NMF Training Loss (All Iterations)')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)

    # Last portion
    n_last = min(50, len(losses))
    ax2.plot(losses[-n_last:])
    ax2.set_xlabel(f'Iteration (last {n_last})')
    ax2.set_ylabel('Total Loss')
    ax2.set_title(f'NMF Training Loss (Last {n_last} checkpoints)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved NMF loss plot to {save_path}")


def plot_nmf_components(H, save_path):
    """Plot the NMF spectral components."""
    K = H.shape[0]
    n_wavelengths = H.shape[1]

    # BOSS wavelength grid
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


def plot_r2_histogram(r2_values, save_path):
    """Plot histogram of R² values for each component."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(r2_values, bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(r2_values), color='red', linestyle='--', label=f'Mean: {np.mean(r2_values):.3f}')
    ax.axvline(np.median(r2_values), color='orange', linestyle='--', label=f'Median: {np.median(r2_values):.3f}')
    ax.set_xlabel('R² (variance explained by labels)')
    ax.set_ylabel('Number of components')
    ax.set_title('How well do stellar labels predict each NMF component?')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved R² histogram to {save_path}")


def plot_residual_histograms(true_labels, inferred_labels, label_names, save_path):
    """Plot histograms of residuals for each label."""
    n_labels = true_labels.shape[1]

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
    print(f"Saved residual histogram plot to {save_path}")


def plot_spectra_comparison(flux_subset, ivar_subset, labels_subset, inferred_labels,
                            H, predict_weights_fn, wavelength, save_path, n_plot=20):
    """Plot observed vs model spectra for a subset of stars."""
    n_subset = min(n_plot, len(flux_subset))

    n_cols = 2
    n_rows = (n_subset + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows), sharex=True)
    axes = axes.ravel()

    for i in range(n_subset):
        ax = axes[i]

        # Observed flux
        obs_flux = flux_subset[i]

        # Model flux from inferred labels
        inferred_weights = predict_weights_fn(inferred_labels[i:i+1])[0]
        model_flux = 1.0 - inferred_weights @ H

        # Model flux from true labels
        true_weights = predict_weights_fn(labels_subset[i:i+1])[0]
        true_model_flux = 1.0 - true_weights @ H

        # Plot
        ax.plot(wavelength, obs_flux, 'k-', lw=0.5, alpha=0.7, label='Observed')
        ax.plot(wavelength, model_flux, 'r-', lw=0.8, alpha=0.8, label='Model (inferred)')
        ax.plot(wavelength, true_model_flux, 'b--', lw=0.8, alpha=0.6, label='Model (true labels)')

        # Labels info
        true_lbl = labels_subset[i]
        inf_lbl = inferred_labels[i]
        ax.set_title(f'Star {i+1}: Teff={true_lbl[0]:.0f}->{inf_lbl[0]:.0f}, '
                     f'logg={true_lbl[1]:.2f}->{inf_lbl[1]:.2f}, '
                     f'[M/H]={true_lbl[2]:.2f}->{inf_lbl[2]:.2f}', fontsize=9)
        ax.set_ylim(0.4, 1.1)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)

    for i in range(n_subset, len(axes)):
        axes[i].set_visible(False)

    if len(axes) >= 2:
        axes[-2].set_xlabel('Wavelength (A)')
        axes[-1].set_xlabel('Wavelength (A)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved spectra comparison to {save_path}")


if __name__ == '__main__':
    # Configuration
    data_file = 'boss_apogee_lux_training_data.npz'
    K = 24  # Reduced number of NMF components (was 128)
    nmf_iterations = 5000
    inference_iterations = 5000

    # Regularization parameters
    l1_alpha = 0.01  # L1 regularization on W (sparsity)
    ortho_alpha = 0.001  # Orthogonality regularization on H

    label_names = ['teff', 'logg', 'm_h', 'alpha_h']

    # Create output directory
    output_dir = 'nmf_results'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving results to: {output_dir}/")

    print("=" * 60)
    print("NMF-based Stellar Spectra Model")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  K = {K} components")
    print(f"  NMF iterations = {nmf_iterations}")
    print(f"  Inference iterations = {inference_iterations}")
    print(f"  L1 regularization = {l1_alpha}")
    print(f"  Orthogonality regularization = {ortho_alpha}")
    print(f"  Model type = Polynomial (bias + linear + quadratic + cross-terms)")

    # Step 1: Load data
    print("\n[1/6] Loading data...")
    absorption, flux, ivar, labels = load_data(data_file)
    n_stars, n_wavelengths = absorption.shape
    print(f"  Loaded {n_stars} stars with {n_wavelengths} wavelength pixels")
    print(f"  Label ranges:")
    for i, name in enumerate(label_names):
        print(f"    {name}: [{labels[:, i].min():.2f}, {labels[:, i].max():.2f}]")

    # Step 2: Perform weighted NMF with regularization
    print(f"\n[2/6] Performing weighted NMF with K={K} components...")
    W, H, nmf_losses = weighted_nmf(
        absorption, ivar, K,
        n_iter=nmf_iterations,
        seed=42,
        l1_alpha=l1_alpha,
        ortho_alpha=ortho_alpha
    )
    print(f"  W shape: {W.shape}, H shape: {H.shape}")

    # Save NMF plots
    plot_nmf_components(H, f'{output_dir}/nmf_components.png')
    if len(nmf_losses) > 1:
        plot_nmf_loss(nmf_losses, f'{output_dir}/nmf_training_loss.png')

    # Step 3: Train label -> weights model (polynomial)
    print("\n[3/6] Training label-to-weights model...")
    theta, label_mean, label_std = train_label_to_weights_linear(labels, W)
    predicted_W = predict_weights_from_labels_linear(labels, theta, label_mean, label_std)

    def predict_weights_fn(labels_input):
        return predict_weights_from_labels_linear(labels_input, theta, label_mean, label_std)

    # Diagnose label -> weights fit
    print("\n  Diagnosing label -> weights fit quality:")
    r2_values, overall_r2 = diagnose_label_to_weights_fit(W, predicted_W, K)
    plot_r2_histogram(r2_values, f'{output_dir}/r2_histogram.png')

    # Step 4: Build forward model
    print("\n[4/6] Forward model built: labels -> weights -> absorption")

    # Step 5: Infer labels for subset
    print("\n[5/6] Inferring stellar parameters...")

    # Select random subset
    np.random.seed(42)
    n_subset = min(1000, len(flux))
    subset_indices = np.random.choice(len(flux), size=n_subset, replace=False)
    flux_subset = flux[subset_indices]
    ivar_subset = ivar[subset_indices]
    labels_subset = labels[subset_indices]

    # JAX-compatible polynomial prediction function
    theta_jnp = jnp.array(theta)

    def jax_build_design_matrix(labels_std):
        """Build design matrix in JAX for a single star (1D input)."""
        # labels_std is shape (4,)
        n_labels = labels_std.shape[0]

        # Bias
        features = [jnp.array([1.0])]

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

    def jax_predict_weights(labels_std):
        """Predict weights from standardized labels using polynomial model."""
        design_vec = jax_build_design_matrix(labels_std)
        return design_vec @ theta_jnp

    inferred_labels = infer_labels_batch(
        flux_subset, ivar_subset, H, jax_predict_weights, label_mean, label_std,
        n_iter=inference_iterations,
        learning_rate=0.01,
        init_labels=labels_subset  # Start from true labels for debugging
    )

    # Step 6: Generate plots
    print("\n[6/6] Generating comparison plots...")

    # Wavelength grid
    loglam = 3.5523 + 0.0001 * np.arange(n_wavelengths)
    wavelength = 10**loglam

    # Spectra comparison
    plot_spectra_comparison(
        flux_subset, ivar_subset, labels_subset, inferred_labels,
        H, predict_weights_fn, wavelength,
        f'{output_dir}/spectra_comparison.png', n_plot=20
    )

    # Label comparison
    plot_comparison(labels_subset, inferred_labels, label_names, f'{output_dir}/label_comparison.png')
    plot_residual_histograms(labels_subset, inferred_labels, label_names, f'{output_dir}/label_residuals.png')

    # Summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    for i, name in enumerate(label_names):
        diff = inferred_labels[:, i] - labels_subset[:, i]
        valid = np.isfinite(diff)
        bias = np.median(diff[valid])
        scatter = np.std(diff[valid])
        mad = np.median(np.abs(diff[valid] - bias))
        print(f"  {name:8s}: bias={bias:+.4f}, scatter={scatter:.4f}, MAD={mad:.4f}")

    # Save results
    print("\nSaving results...")
    np.savez(f'{output_dir}/nmf_model_results.npz',
             W=W, H=H,
             theta=theta,
             label_mean=label_mean, label_std=label_std,
             true_labels=labels,
             subset_indices=subset_indices,
             inferred_labels=inferred_labels,
             r2_values=r2_values,
             overall_r2=overall_r2,
             nmf_losses=nmf_losses)

    print(f"  Saved to {output_dir}/")

    print("\nDone!")
