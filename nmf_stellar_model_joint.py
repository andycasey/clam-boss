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
import jax.nn as jnn
from jax import jit, vmap
import optax
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import warnings
from tqdm import tqdm, trange
from sklearn.decomposition import NMF
import configparser
import shutil
import logging

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
    labels_std_batch = jnp.atleast_2d(labels_std_batch)
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
                       print_every=500, seed=42, label_weight=1.0,
                       per_label_weights=None,
                       label_mean=None, label_std=None,
                       scatter=None,
                       theta=None, H=None):
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

    logger.info(f"Joint optimization setup:")
    logger.info(f"  Stars: {n_stars}, Wavelengths: {n_wavelengths}")
    logger.info(f"  K = {K} components")
    logger.info(f"  Design matrix features: {n_features}")
    logger.info(f"  Total parameters: {n_stars * n_labels + n_features * K + K * n_wavelengths:,}")

    # Standardize labels
    if label_mean is None:
        label_mean = np.nanmean(init_labels, axis=0)
    if label_std is None:
        label_std = np.nanstd(init_labels, axis=0)
    init_labels_std = (init_labels - label_mean) / label_std

    # Initialize NMF from absorption spectra
    absorption = np.clip(1.0 - flux, 0.0, np.inf)
    W_init, H_init = initialize_nmf(absorption, K, seed=seed, max_iter=200)
    #W_init = np.random.uniform(0, 1, size=W_init.shape)
    #H_init = np.random.uniform(0, 1, size=H_init.shape)

    # sets nan to label means
    init_labels_std_no_nan = init_labels_std.copy()
    nan_mask = np.isnan(init_labels_std_no_nan)
    init_labels_std_no_nan[nan_mask] = np.take(label_mean, np.where(nan_mask)[1])
    # auto downweight all nanmask
    if per_label_weights is None:
        per_label_weights_jnp = jnp.zeros_like(init_labels_std_jnp) + 1.
    else:
        per_label_weights_jnp = jnp.array(per_label_weights)
    per_label_weights_jnp = jnp.where(nan_mask, 0., per_label_weights_jnp)

    # Initialize theta from initial W and labels
    design_matrix = build_design_matrix_np(init_labels_std_no_nan)
    raw_target = np.log(np.expm1(np.maximum(W_init, 1e-8)))
    theta_init, _, _, _ = np.linalg.lstsq(design_matrix, raw_target, rcond=None)



    # Convert to JAX arrays
    flux_jnp = jnp.array(flux)
    var_jnp = 1.0/jnp.maximum(ivar, 1e-16)
    label_mean_jnp = jnp.array(label_mean)
    label_std_jnp = jnp.array(label_std)
    init_labels_std_jnp = jnp.array(init_labels_std_no_nan)

    # Parameters to optimize (all unconstrained, we'll apply constraints in forward pass)
    # Use log-space for H to ensure positivity
    if theta is not None:
        theta_init = theta
    if H is not None:
        H_init = H
    if scatter is not None:
        ln_scatter_init = jnp.log(scatter)
    else:
        ln_scatter_init = 0.1 * jnp.ones(n_wavelengths) # initialize scatter params
    params = {
        'labels_std': jnp.array(init_labels_std_no_nan),
        'theta': jnp.array(theta_init),
        'log_H': jnp.log(jnp.array(H_init) + 1e-10),
        'ln_scatter': ln_scatter_init
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
        W = jnn.softplus(design_matrix @ theta)

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
        # only count those that contribute to loss
        ncontrib = jnp.sum(per_label_weights_jnp > 0)
        label_loss = jnp.nansum(per_label_weights_jnp * (label_residual ** 2)) / ncontrib # (n_stars * n_labels)

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
    logger.info(f"  Initial loss: {initial_loss:.6e}")

    # Optimization loop
    losses = []
    logger.info(f"\nOptimizing for {n_iter} iterations...")

    with tqdm(total=n_iter) as pb:

        for i in range(n_iter):
            params, opt_state, loss = update_step(params, opt_state)

            #if (i + 1) % print_every == 0:
            losses.append(float(loss))
            pb.set_description(f"loss = {float(loss):.4e}")
            pb.update()

    final_loss = float(loss_fn(params))
    logger.info(f"  Final loss: {final_loss:.6e}")

    # Extract final parameters
    labels_std_final = np.array(params['labels_std'])
    labels_final = labels_std_final * label_std + label_mean
    theta_final = np.array(params['theta'])
    H_final = np.exp(np.array(params['log_H']))
    scatter = np.exp(np.array(params['ln_scatter']))

    # Compute final W
    design_matrix_final = build_design_matrix_np(labels_std_final)
    W_final = np.array(jnn.softplus(design_matrix_final @ theta_final))

    return labels_final, theta_final, H_final, W_final, label_mean, label_std, losses, scatter


def joint_optimization_with_em(
    flux, ivar,
    labels_obs_phys,        # shape (n_stars, n_labels), put arbitrary values where missing
    labels_obs_mask,        # bool mask (n_stars, n_labels): True = observed/fixed, False = missing
    theta, H, K,
    label_mean, label_std,
    scatter,
    infer_kwargs=None,      # dict of kwargs forwarded to infer_labels()
    joint_kwargs=None,      # dict of kwargs forwarded to your existing joint_optimization()
    em_iters=3,
    em_adam_iters=200,
):
    """
    Wrapper that runs EM-style training:
      - E-step: infer missing labels with infer_labels(..., fixed_mask=~labels_obs_mask)
      - M-step: call your original joint_optimization(...) with filled labels

    Minimal changes: original joint_optimization() is called unchanged.
    Returns: theta, H, labels_filled_phys
    """

    if infer_kwargs is None:
        infer_kwargs = {}
    if joint_kwargs is None:
        joint_kwargs = {}

    n_stars = labels_obs_phys.shape[0]
    n_labels = labels_obs_phys.shape[1]

    # make a working copy of labels (physical units). For missing entries, fill with label_mean as a safe init
    labels_filled = labels_obs_phys.copy().astype(float)
    missing = ~np.asarray(labels_obs_mask).astype(bool)
    if np.any(missing):
        # initialize missing entries to global mean (or you can use other init)
        for i in range(n_stars):
            for j in range(n_labels):
                if missing[i, j]:
                    labels_filled[i, j] = label_mean[j]

    # Prepare fixed_mask to pass into infer_labels: True=fix, False=free
    # Note infer_labels expects fixed_mask meaning "fix this dim", but we wrote infer with fixed_mask where True=fix.
    fixed_mask_for_infer = np.asarray(labels_obs_mask).astype(bool)  # True where observed -> fix

    # Use standardized initial labels for infer_labels init if it helps convergence
    def phys_to_std(arr_phys, label_mean, label_std):
        return (np.asarray(arr_phys) - label_mean) / label_std

    # EM loop
    for em_it in range(em_iters):
        # --- E-step: infer missing dims using current theta/H
        # Provide current labels_filled as the init; infer_labels expects init_labels_std (standardized)
        init_labels_std = phys_to_std(labels_filled, label_mean, label_std)

        # call infer_labels to re-fit missing dims. We pass fixed_mask so it will optimize only missing dims.
        # Note: infer_labels' fixed_mask param means "True => fix this dim", so we pass exactly labels_obs_mask
        infer_args = dict(
            flux=flux,
            ivar=ivar,
            theta=theta,
            H=H,
            label_mean=label_mean,
            label_std=label_std,
            scatter=scatter,
            init_labels_std=init_labels_std,
            n_iter=em_adam_iters,
            learning_rate=infer_kwargs.get('learning_rate', 0.05),
            optimizer=infer_kwargs.get('optimizer', 'adam'),
            fixed_mask=fixed_mask_for_infer,
            **{k: v for k, v in infer_kwargs.items() if k not in ['learning_rate', 'optimizer']}
        )
        # infer_labels returns labels in physical units (per our implementation)
        labels_new = infer_labels(**infer_args)  # shape (n_stars, n_labels) physical

        # Replace only missing entries in labels_filled (observed entries stay as observed)
        labels_filled[missing] = labels_new[missing]

        # --- M-step: call your original joint_optimization to update theta, H using labels_filled
        labels_filled, theta, H, W, label_mean, label_std, losses, scatter = joint_optimization(
            flux, ivar, labels_filled, K,
            label_mean=label_mean, label_std=label_std, scatter=scatter,
            theta=theta, H=H,
            **joint_kwargs
        )
    return labels_filled, theta, H, W, label_mean, label_std, losses, scatter


def compute_nmf_weights(flux, H):
    """Compute W from flux using least squares."""
    
    # Convert to absorption
    absorption = np.clip(1.0 - flux, 0.0, np.inf)
    
    # Solve: W @ H = absorption
    # W = absorption @ H.T @ (H @ H.T)^{-1}
    H_HT_inv = np.linalg.inv(H @ H.T + 1e-6 * np.eye(H.shape[0]))
    W = absorption @ H.T @ H_HT_inv
    
    # Enforce non-negativity
    W = np.maximum(W, 0.0)
    
    return W


def save_ridge_model_npz(ridge_model, W_scaler, label_scaler, 
                         save_path='nmf_ridge_model.npz'):
    """
    Save Ridge model as numpy arrays in NPZ format.
    """
    logger.info(f"Saving Ridge model to {save_path}...")
    
    # Extract the actual parameters (just numpy arrays!)
    model_params = {
        # Ridge model parameters
        'ridge_coef': ridge_model.coef_,           # (n_labels, K)
        'ridge_intercept': ridge_model.intercept_,  # (n_labels,)
        
        # W scaler parameters
        'W_mean': W_scaler.mean_,                   # (K,)
        'W_scale': W_scaler.scale_,                 # (K,)
        
        # Label scaler parameters
        'label_mean': label_scaler.mean_,           # (n_labels,)
        'label_scale': label_scaler.scale_,         # (n_labels,)
    }
    
    np.savez(save_path, **model_params)
    return


def train_and_save_ridge_model(W_train, train_labels, save_path='nmf_ridge_model.npz',
                               alpha=1.0):
    """
    Train and save ridge model used for initial guess
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    
    logger.info("Training Ridge regression...")
    
    # Fit scalers
    W_scaler = StandardScaler()
    label_scaler = StandardScaler()
    
    W_train_scaled = W_scaler.fit_transform(W_train)
    labels_scaled = label_scaler.fit_transform(train_labels)
    
    # Train Ridge
    ridge = Ridge(alpha=alpha)
    ridge.fit(W_train_scaled, labels_scaled)
    
    # save model
    save_ridge_model_npz(ridge, W_scaler, label_scaler, save_path)
    
    logger.info(f"\nModel saved to {save_path}")
    
    return


def load_ridge_model_npz(model_path='nmf_ridge_model.npz'):
    """
    Load Ridge model from NPZ file.
    """
    data = np.load(model_path)
    
    model_params = {
        'ridge_coef': data['ridge_coef'],
        'ridge_intercept': data['ridge_intercept'],
        'W_mean': data['W_mean'],
        'W_scale': data['W_scale'],
        'label_mean': data['label_mean'],
        'label_scale': data['label_scale'],
    }
    
    return model_params


def predict_with_ridge_npz(W, model_params):
    """
    Predict labels using saved Ridge model parameters.

    Used for initial guess of optimization
    """
    # Standardize W
    W_scaled = (W - model_params['W_mean']) / model_params['W_scale']
    
    # Predict (Ridge is just linear: y = X @ coef + intercept)
    labels_scaled = W_scaled @ model_params['ridge_coef'].T + model_params['ridge_intercept']
    
    # Inverse transform labels
    predicted_labels = labels_scaled * model_params['label_scale'] + model_params['label_mean']
    
    return predicted_labels


def infer_labels(flux, ivar, theta, H, label_mean, label_std, scatter, init_labels_std=None,
                 n_iter=2000, learning_rate=0.05, seed=42, optimizer='adam',
                 grid_points=5, grid_range=(-3.0, 3.0),
                 fixed_mask=None, fixed_values_phys=None):
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
    n_iter : int | list
        Number of optimization iterations (for Adam) or max iterations (for BFGS).
        Can be list for two-stage optimization, which then sets for each step.
    learning_rate : float
        Learning rate for Adam optimizer (ignored for BFGS)
    seed : int
        Random seed
    optimizer : str
        Optimization method: 'adam' or 'bfgs' or 'two-stage'
    grid_points : int | list
        Number of grid points per dimension for initial grid search
        (only used when init_labels_std is None)
    grid_range : tuple | list
        (min, max) range in standardized coordinates for grid search
    fixed_mask : None, or array-like
        Optional. If provided, can be shape (n_labels,) or (n_stars, n_labels).
        True means the corresponding label dimension is fixed (not optimized).
    fixed_values_phys : None, or array-like
        Optional. Physical-unit values to fix to when fixed_mask is True.
        Broadcast rules same as fixed_mask.

    Returns:
    --------
    labels : array (n_stars, 4)
        Inferred stellar labels (teff, logg, m_h, alpha_h)
    """
    from scipy.optimize import minimize
    from itertools import product

    n_stars, n_wavelengths = flux.shape
    n_labels = len(label_mean)

    logger.info(f"Inferring labels for {n_stars} stars using {optimizer.upper()}...")

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
        W = jnn.softplus(design_vec @ theta_jnp)
        pred_flux = 1.0 - W @ H_jnp
        total_var = var_single + scatter_sq
        chi_sq = (flux_single - pred_flux)**2 / total_var
        return 0.5 * jnp.sum(chi_sq)

    single_star_loss_and_grad = jit(jax.value_and_grad(single_star_loss))

    # Grid search to find initial values if not provided
    if init_labels_std is None:
        # Build the grid once (shared across all stars)
        if isinstance(grid_range, tuple):
            logger.info(f"  Performing batched grid search ({grid_points}^{n_labels} = {grid_points**n_labels} points per star)...")
            grid_1d = jnp.linspace(grid_range[0], grid_range[1], grid_points)
            grid_points_all = jnp.array(list(product(*[grid_1d]*n_labels)))  # (n_grid, n_labels)
        else:
            logger.info(f"  Performing batched grid search {np.prod(grid_points)} points per star)...")
            grid_1d = [jnp.linspace(grid_range[i][0], grid_range[i][1], grid_points[i]) for i in range(len(grid_range))]
            grid_points_all = jnp.array(list(product(*grid_1d)))
        n_grid = len(grid_points_all)
        
        logger.info(f"    Grid has {n_grid:,} points")
        
        # Define the grid search function for a single star
        @jit
        def grid_search_star(flux_single, var_single):
            """Find best grid point for a single star."""
            losses = vmap(lambda grid_pt: single_star_loss(grid_pt, flux_single, var_single))(
                grid_points_all
            )
            best_idx = jnp.argmin(losses)
            return grid_points_all[best_idx]
        
        # Vectorize across a batch of stars
        grid_search_batch = jit(vmap(grid_search_star))
        
        # Process stars in batches to avoid memory issues
        batch_size = 500  # Adjust based on your GPU memory
        n_batches = (n_stars + batch_size - 1) // batch_size
        init_labels_std = np.zeros((n_stars, n_labels))
        
        logger.info(f"    Processing {n_stars} stars in {n_batches} batches of ~{batch_size}...")
        
        for i in tqdm(range(n_batches), desc="Grid search batches"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_stars)
            
            # Process this batch
            batch_result = grid_search_batch(
                flux_jnp[start_idx:end_idx],
                var_jnp[start_idx:end_idx]
            )
            init_labels_std[start_idx:end_idx] = np.array(batch_result)
        
        logger.info(f"  Grid search complete.")

    # add option for when some are fixed during EM-style joint opt
    if fixed_mask is not None:
        # Normalize fixed_mask to shape (n_stars, n_labels)
        fm = np.asarray(fixed_mask)
        if fm.ndim == 1:
            if fm.size != n_labels:
                raise ValueError("fixed_mask length must equal n_labels")
            fm = np.tile(fm[None, :], (n_stars, 1))
        elif fm.shape != (n_stars, n_labels):
            raise ValueError("fixed_mask must be shape (n_stars, n_labels) or (n_labels,)")

        # fixed_values_phys optional: broadcast if needed and convert to standardized
        if fixed_values_phys is not None:
            fv = np.asarray(fixed_values_phys)
            if fv.ndim == 1:
                if fv.size != n_labels:
                    raise ValueError("fixed_values_phys length must equal n_labels")
                fv = np.tile(fv[None, :], (n_stars, 1))
            elif fv.shape != (n_stars, n_labels):
                raise ValueError("fixed_values_phys must be shape (n_stars, n_labels) or (n_labels,)")
            fv_std = (fv - label_mean) / (label_std + 1e-12)
        else:
            fv_std = None

        # Convert to jnp arrays for jit/vmap
        init_std_jnp = jnp.array(init_labels_std)
        fv_std_jnp = jnp.array(fv_std) if fv_std is not None else None
        fm_bool = np.array(fm, dtype=bool)

        # Group stars by unique mask pattern
        mask_tuples = [tuple(row.tolist()) for row in fm_bool]
        unique_masks = {}
        for idx, m in enumerate(mask_tuples):
            unique_masks.setdefault(m, []).append(idx)

        labels_std_out = np.zeros((n_stars, n_labels))

        # For each unique mask, optimize in batch using Adam
        for mask_tuple, indices in unique_masks.items():
            idx_arr = np.array(indices, dtype=int)
            m_bool = np.array(mask_tuple, dtype=bool)
            free_idx = np.where(~m_bool)[0]
            k = free_idx.size

            # If no free dims, just fill from fixed values or init
            if k == 0:
                if fv_std is not None:
                    labels_std_out[idx_arr] = np.array(fv_std_jnp[idx_arr])
                else:
                    labels_std_out[idx_arr] = np.array(init_std_jnp[idx_arr])
                continue

            # Prepare per-group arrays (jnp)
            init_full_group_jnp = jnp.array(init_std_jnp[idx_arr])   # (gsize, n_labels)
            z0_group = jnp.array(init_full_group_jnp[:, free_idx])   # (gsize, k)
            flux_group_jnp = jnp.array(flux_jnp[idx_arr])            # (gsize, n_lambda)
            var_group_jnp = jnp.array(var_jnp[idx_arr])              # (gsize, n_lambda)

            if fv_std is not None:
                fixed_vals_for_solver_jnp = jnp.array(fv_std_jnp[idx_arr])  # (gsize, n_labels)
            else:
                fixed_vals_for_solver_jnp = None

            # Define per-star loss fun: inputs z (k,), init_full (n_labels,), fixed_vals (n_labels,), flux,var
            def loss_per_star(z, init_full, fixed_vals, flux_s, var_s):
                full = init_full
                full = full.at[free_idx].set(z)
                if fixed_vals is not None:
                    full = full.at[m_bool].set(fixed_vals[m_bool])
                return single_star_loss(full, flux_s, var_s)

            # Vectorized grad fn: returns grads shape (gsize, k)
            grad_fn = jax.jit(jax.vmap(jax.grad(loss_per_star, argnums=0),
                                       in_axes=(0, 0, 0, 0, 0)))

            # Adam optimizer setup for z (per-group, batched)
            lr = float(learning_rate)
            opt = optax.adam(lr)
            opt_state = opt.init(z0_group)
            z = z0_group  # jnp array shape (gsize, k)

            # update step (jit)
            @jax.jit
            def adam_step(z, opt_state, init_full_group, fixed_vals_group, flux_group, var_group):
                grads = grad_fn(z, init_full_group, fixed_vals_group, flux_group, var_group)  # (g,k)
                updates, opt_state = opt.update(grads, opt_state, z)
                z = optax.apply_updates(z, updates)
                return z, opt_state

            # run optimization for n_iter steps (you can tune n_iter for speed)
            # If n_iter is a list in two-stage, pick appropriate stage length. Here use int.
            nit = int(n_iter) if not isinstance(n_iter, (list, tuple)) else int(n_iter[0])
            for it in trange(nit):
                z, opt_state = adam_step(z, opt_state, init_full_group_jnp, fixed_vals_for_solver_jnp, flux_group_jnp, var_group_jnp)

            # Reconstruct full standardized labels for this group
            z_np = np.array(z)  # (gsize, k)
            full_group = np.array(init_full_group_jnp)  # numpy copy
            for ii in range(len(idx_arr)):
                full_group[ii, free_idx] = z_np[ii]
                if fv_std is not None:
                    full_group[ii, m_bool] = np.array(fv_std_jnp[idx_arr[ii]])[m_bool]
            labels_std_out[idx_arr] = full_group

        # Convert to physical and return
        labels_final = labels_std_out * label_std + label_mean
        return labels_final

    if optimizer == 'two-stage' or optimizer == 'adam':
        # do adam and save as init if two-stage warmup
        params = {'labels_std': jnp.array(init_labels_std)}

        @jit
        def forward(params):
            """Compute predicted flux from labels."""
            labels_std = params['labels_std']
            design_matrix = build_design_matrix_batch_jax(labels_std)
            W = jnn.softplus(design_matrix @ theta_jnp)
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

        if optimizer == 'adam':
            n_iteri = n_iter
        else:
            n_iteri = n_iter[0]
        with tqdm(total=n_iteri) as pb:
            for i in range(n_iteri):
                params, opt_state, loss = update_step(params, opt_state)
                pb.set_description(f"loss = {float(loss):.4e}")
                pb.update()

        if optimizer == 'adam':
            labels_std_final = np.array(params['labels_std'])
        else:
            init_labels_std = np.array(params['labels_std'])  # update to results from adam for two-stage
    if optimizer == 'two-stage' or optimizer == 'bfgs':
        # GPU-accelerated BFGS using jaxopt
        try:
            import jaxopt
        except ImportError:
            raise ImportError("Install jaxopt: pip install jaxopt")
        
        logger.info(f"  Running L-BFGS-B optimization (GPU-accelerated)...")
        labels_std_final = np.zeros((n_stars, n_labels))
        
        # Create solver
        if optimizer == 'bfgs':
            n_iteri = n_iter
        else:
            n_iteri = n_iter[1]
        solver = jaxopt.LBFGS(fun=single_star_loss, maxiter=n_iteri, tol=1e-6)
        
        # Batch size - adjust based on GPU memory (100 is safe for most GPUs)
        batch_size = 100
        n_batches = (n_stars + batch_size - 1) // batch_size
        
        @jit
        def optimize_batch(init_batch, flux_batch, var_batch):
            def optimize_single(init_single, flux_single, var_single):
                result = solver.run(init_single, flux_single, var_single)
                return result.params
            return vmap(optimize_single)(init_batch, flux_batch, var_batch)
        
        for i in tqdm(range(n_batches), desc="BFGS batches"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_stars)
            
            batch_result = optimize_batch(
                jnp.array(init_labels_std[start_idx:end_idx]),
                flux_jnp[start_idx:end_idx],
                var_jnp[start_idx:end_idx]
            )
            labels_std_final[start_idx:end_idx] = np.array(batch_result)

    else:  # adam
        # Batch Adam optimization (original behavior)
        params = {'labels_std': jnp.array(init_labels_std)}

        @jit
        def forward(params):
            """Compute predicted flux from labels."""
            labels_std = params['labels_std']
            design_matrix = build_design_matrix_batch_jax(labels_std)
            W = jnn.softplus(design_matrix @ theta_jnp)
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


def plot_test_comparison(true_labels, inferred_labels,
                         label_names, save_path,
                         label_bounds={
                            'teff': (2500, 20000),
                            'logg': (0.5, 5.5),
                            'm_h': (-4., 0.75),
                            'alpha_h': (-0.5, 0.6)
                         }):
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
    logger.info(f"Saved test comparison plot to {save_path}")


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


def plot_comparison(true_labels, inferred_labels,
                    label_names, save_path,
                    label_bounds = {
                        'teff': (2500, 20000),
                        'logg': (0.5, 5.5),
                        'm_h': (-4., 0.75),
                        'alpha_h': (-0.5, 0.6)
                    }):
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
    logger.info(f"Saved comparison plot to {save_path}")


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
    logger.info(f"Saved NMF components plot to {save_path}")


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
    logger.info(f"Saved loss plot to {save_path}")


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
    logger.info(f"Saved spectra comparison to {save_path}")


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
    logger.info(f"Saved residual histogram to {save_path}")


def kiel_diagram(y_test: np.ndarray,
                 predictions: np.ndarray,
                 label_names: list,
                 save_dir: str,
                 fe_h=False,
                 teff_max=20000,
                 feh_min=-3):
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
        plt.savefig(f'{save_dir}/kiel_diagram_fe_h.png')
    else:
        plt.savefig(f'{save_dir}/kiel_diagram.png')
    plt.close()


def alpha_fe_plot(y_test: np.ndarray,
                  predictions: np.ndarray,
                  label_names: list,
                  save_dir: str,
                  convert_alpha: bool,
                  feh_min: float = -3.):
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
    plt.savefig(f'{save_dir}/alpha_m_vs_fe_h.png')
    plt.close()



def unf_training_sample(true_labels, nbins,
                        nstars_per_bin,
                        random_seed=None, ranges=None):
    """
    Create a uniform sampling across parameter space using multi-dimensional binning.
    
    Parameters
    ----------
    true_labels : array-like, shape (N, D)
        The parameter values for N samples across D dimensions
    nbins : int
        Number of bins per dimension
    nstars_per_bin : int
        Target number of stars to sample per bin
    random_seed : int, optional
        Random seed for reproducibility
    ranges: list
        set the range for the dimensions
        
    Returns
    -------
    indices : array
        Indices of selected samples from the original true_labels array
    """
    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)

    N, D = true_labels.shape

    if ranges is None:
        ranges = [None] * D
    
    # Create bin edges for each dimension
    bin_edges = []
    for d in range(D):
        if ranges[d] is None:
            edges = np.linspace(true_labels[:, d].min(), 
                               true_labels[:, d].max(), 
                               nbins + 1)
        else:
            edges = np.linspace(ranges[d][0], 
                                ranges[d][1], 
                                nbins + 1)
        bin_edges.append(edges)
    
    # Assign each sample to a bin (multi-dimensional bin index)
    bin_indices = np.zeros((N, D), dtype=int)
    for d in range(D):
        # digitize returns 1-indexed bins, so subtract 1
        bin_indices[:, d] = np.digitize(true_labels[:, d], bin_edges[d]) - 1
        # Handle edge case where max values get assigned to bin nbins
        bin_indices[:, d] = np.clip(bin_indices[:, d], 0, nbins - 1)
    
    # Collect samples from each occupied bin
    selected_indices = []
    
    # Iterate through all possible bin combinations
    for bin_combo in np.ndindex(*([nbins] * D)):
        # Find all samples in this bin
        mask = np.all(bin_indices == bin_combo, axis=1)
        samples_in_bin = np.where(mask)[0]
        
        if len(samples_in_bin) == 0:
            continue
        
        # Sample nstars_per_bin or all available stars
        n_to_sample = min(nstars_per_bin, len(samples_in_bin))
        sampled = np.random.choice(samples_in_bin, size=n_to_sample, replace=False)
        selected_indices.extend(sampled)
    
    return np.array(selected_indices)



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
        output_dir = f'nmf_joint_results_with_scatter_K32{append_wb}{append_ms}{append_hs}'
    else:
        label_names = ['teff', 'logg', 'm_h', 'alpha_m']
        output_dir = f'nmf_joint_results_with_scatter_K32_alpha_m{append_wb}{append_ms}{append_hs}'

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
    data_file = 'boss_apogee_lux_training_data.npz'
    absorption, flux, ivar, true_labels = load_data(data_file,
                                                    convert_alpha=convert_alpha)

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

        # remove nans
        labels_mask = np.isfinite(true_labels)
        keep_stars = np.all(labels_mask, axis=1)
        absorption = absorption[keep_stars]
        flux = flux[keep_stars]
        ivar = ivar[keep_stars]
        true_labels = true_labels[keep_stars]
    
    if add_HS:
        # load HSs, append later
        absorption_hs, flux_hs, ivar_hs, true_labels_hs = load_data(
            data_file_hs,
            convert_alpha=convert_alpha)
    
    n_stars, n_wavelengths = flux.shape
    logger.info(f"  Loaded {n_stars} stars with {n_wavelengths} wavelength pixels")

    # Joint optimization
    logger.info("\n[2/4] Running joint optimization...")
    # get training subsample
    if train_w_subsample:
        if add_MS:
            nbins = 25
            nstars_per_bin = 4
            ranges = [[3000, 6500],
                      [1, 5.25],
                      [-3.5, 0.5],
                      [-0.1, 0.5]]
        elif add_WBs and remove_nans:
            nbins = 16
            nstars_per_bin = 7
            ranges = [[3000, 6500],
                      [1, 5.25],
                      [-2, 0.5],
                      [-0.1, 0.3]]
        else:
            nbins = 15
            nstars_per_bin = 7
            ranges = [[4000, 6500],
                      [1, 5],
                      [-2, 0.5],
                      [-0.1, 0.3]]
        idx_train = unf_training_sample(
            true_labels,
            nbins, nstars_per_bin,
            random_seed=42, ranges=ranges)
    else:
        idx_train = np.arange(len(true_labels))
    if add_WBs and not remove_nans:
        logger.info("\nRunning EM-style joint optimization...")
        absorption = np.append(absorption, absorption_wb, axis=0)
        flux = np.append(flux, flux_wb, axis=0)
        ivar = np.append(ivar, ivar_wb, axis=0)
        true_labels = np.append(true_labels, true_labels_wb, axis=0)
        labels_mask = np.isfinite(true_labels)
        init_stars = np.all(labels_mask, axis=1)

        # add some WBs to training set
        rng = np.random.default_rng(seed=42)
        id_bins = np.arange(n_stars, len(flux), 1)
        idx_train_bin = rng.choice(
            id_bins,
            size=int(len(id_bins) * 0.8), replace=False)
        
        idx_train = np.append(idx_train, idx_train_bin)

        if add_HS:
            # add in the hot stars at end
            absorption = np.append(absorption, absorption_hs, axis=0)
            flux = np.append(flux, flux_hs, axis=0)
            ivar = np.append(ivar, ivar_hs, axis=0)
            true_labels = np.append(true_labels, true_labels_hs, axis=0)
            labels_mask = np.isfinite(true_labels)
            init_stars = np.all(labels_mask, axis=1)

            # add some hotstars to training set
            rng = np.random.default_rng(seed=42)
            id_bins = np.arange(n_stars, len(flux), 1)
            idx_train_bin = rng.choice(
                id_bins,
                size=int(len(id_bins) * 0.8), replace=False)
            
            idx_train = np.append(idx_train, idx_train_bin)

        per_label_weights = np.zeros_like(true_labels) + 1.
        per_label_weights[~labels_mask] = 0.01  # little wait to missing, mostly learned

        # start with non-nan
        label_mean0 = np.nanmean(true_labels[idx_train], axis=0)
        label_std0 = np.nanstd(true_labels[idx_train], axis=0)
        init_true_labels = true_labels.copy()
        init_true_labels[~labels_mask] = 4.5  # for now this assumes logg missing
        inferred_labels0, theta0, H0, W0, label_mean0, label_std0, losses0, scatter0 = joint_optimization(
            flux[idx_train],
            ivar[idx_train],
            init_true_labels[idx_train],
            K,
            per_label_weights=per_label_weights[idx_train],
            n_iter=5_000,
            learning_rate=learning_rate,
            print_every=print_every,
            seed=42,
            label_mean=label_mean0,
            label_std=label_std0
        )

        # now do EM
        inferred_labels, theta, H, W, label_mean, label_std, losses, scatter = joint_optimization_with_em(
            flux[idx_train], ivar[idx_train],
            true_labels[idx_train],
            labels_mask[idx_train],
            theta0, H0, K,
            label_mean0, label_std0,
            scatter0,
            infer_kwargs={'learning_rate': 0.01},
            joint_kwargs={'n_iter': 3_000,
                        'learning_rate': learning_rate,
                        'print_every': print_every,
                        'seed': 42,
                        'per_label_weights': per_label_weights[idx_train]},
            em_iters=4,
            em_adam_iters=500,
        )
    else:
        logger.info("\nRunning nominal joint optimization...")

        if add_HS:
            # add in the hot stars at end
            absorption = np.append(absorption, absorption_hs, axis=0)
            flux = np.append(flux, flux_hs, axis=0)
            ivar = np.append(ivar, ivar_hs, axis=0)
            true_labels = np.append(true_labels, true_labels_hs, axis=0)
            labels_mask = np.isfinite(true_labels)
            init_stars = np.all(labels_mask, axis=1)

            # add some WBs to training set
            rng = np.random.default_rng(seed=42)
            id_bins = np.arange(n_stars, len(flux), 1)
            idx_train_bin = rng.choice(
                id_bins,
                size=int(len(id_bins) * 0.8), replace=False)
            
            idx_train = np.append(idx_train, idx_train_bin)

            per_label_weights = np.zeros_like(true_labels) + 1.
            per_label_weights[~labels_mask] = 0.0
        else:
            per_label_weights = np.zeros_like(true_labels) + 1.
    
        inferred_labels, theta, H, W, label_mean, label_std, losses, scatter = joint_optimization(
            flux[idx_train], ivar[idx_train], true_labels[idx_train], K,
            n_iter=n_iter,
            learning_rate=learning_rate,
            print_every=print_every,
            seed=42,
            per_label_weights=per_label_weights[idx_train]
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
                               save_path=model_path, alpha=1.)
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
    test_inferred_labels = infer_labels(
        test_flux, test_ivar,
        theta, H, label_mean, label_std, scatter,
        init_labels_std=init_labels_std,
        n_iter=[100, 1000],
        learning_rate=0.01,
        optimizer='two-stage',
        grid_points=None,
        grid_range=None
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

    # Save test results
    np.savez(f'{output_dir}/test_inference_results.npz',
             test_true_labels=test_true_labels,
             test_inferred_labels=test_inferred_labels,
             test_stats=test_stats)
    logger.info(f"Saved test results to {output_dir}/test_inference_results.npz")

    logger.info("\nDone!")
