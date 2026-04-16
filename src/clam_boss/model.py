import numpy as np
import jax
import jax.numpy as jnp
import jax.nn as jnn
from jax import jit, vmap, pmap
import optax
import warnings
from tqdm import tqdm, trange
from sklearn.decomposition import NMF
import logging
from typing import Tuple
try:
    from jaxlib._jax import ArrayImpl
except ModuleNotFoundError:
    from jaxlib.xla_extension import ArrayImpl

jax.config.update("jax_enable_x64", True)


def initialize_nmf(absorption: np.ndarray,
                   K: int,
                   seed: int = 42,
                   max_iter: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initialize NMF using sklearn for a good starting point.

    Paramaters
    ----------
    absorption: np.ndarray
        absoption spectra for the model
    
    K: int
        number of components for NMF
    
    seed: int
        random seeing
    
    max_iter: int
        maximum number of iterations
    
    Returns
    -------
    W_init: np.ndarray
        Weights from the resulting NMF
    
    H_init: np.ndarray
        basis vector maxtrix from the resulting NMF
    """
    model = NMF(n_components=K, init='nndsvda', max_iter=max_iter, random_state=seed)
    W_init = model.fit_transform(absorption)
    H_init = model.components_
    return W_init, H_init


def build_design_matrix_jax(labels_std: ArrayImpl) -> ArrayImpl:
    """
    Build design matrix for a single star in JAX.

    For 4 labels, creates 15 features:
    - 1 bias term
    - 4 linear terms
    - 4 quadratic terms
    - 6 cross-terms

    Parameters
    ----------
    labels_std: jaxlib._jax.ArrayImpl
        Jax array of the labels for the stars
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


def joint_optimization(flux: np.ndarray,
                       ivar: np.ndarray,
                       init_labels: np.ndarray,
                       K: int,
                       n_iter: int = 5000,
                       learning_rate: float = 0.01,
                       print_every: int = 500,
                       seed: int = 42,
                       label_weight: float = 1.0,
                       per_label_weights: np.ndarray = None,
                       label_mean: np.ndarray = None,
                       label_std: np.ndarray = None,
                       scatter: np.ndarray = None,
                       theta: np.ndarray = None,
                       H: np.ndarray = None,
                       logger=None):
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
    print_every: int
        How often to print loss
    seed: int
        random seed
    label_weight : float
        Weight for label loss term (penalizes deviation from initial labels)
    per_label_weights: array (n_stars, 4)
        Give a weight to each label (teff, logg, m_h, alpha_h) for each star.
        Option, will == 1 if None
    label_mean: array
        Can fix means of labels, if None calculated in function
    label_std: array
        Can fix stds of labels, if None calculated in function
    scatter: array
        initial value for scatter, if None then uses good initial guess
    theta: array
        initial value for theta, if None then uses good initial guess
    H: array
        initial value for H, if None then uses good initial guess
    logger: logging.logger
        If not None, send print out to log
    
    Returns:
    --------
    labels : optimized stellar labels
    theta : polynomial coefficients (15 x K)
    H : NMF basis vectors (K x n_wavelengths)
    """
    n_stars, n_wavelengths = flux.shape
    n_labels = init_labels.shape[1]
    n_features = 1 + n_labels + n_labels + n_labels * (n_labels - 1) // 2  # 15

    if logger is not None:
        logger.info(f"Joint optimization setup:")
        logger.info(f"  Stars: {n_stars}, Wavelengths: {n_wavelengths}")
        logger.info(f"  K = {K} components")
        logger.info(f"  Design matrix features: {n_features}")
        logger.info(f"  Total parameters: {n_stars * n_labels + n_features * K + K * n_wavelengths:,}")
    else:
        print(f"Joint optimization setup:")
        print(f"  Stars: {n_stars}, Wavelengths: {n_wavelengths}")
        print(f"  K = {K} components")
        print(f"  Design matrix features: {n_features}")
        print(f"  Total parameters: {n_stars * n_labels + n_features * K + K * n_wavelengths:,}")

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
    if logger is not None:
        logger.info(f"  Initial loss: {initial_loss:.6e}")

    # Optimization loop
    losses = []
    if logger is not None:
        logger.info(f"\nOptimizing for {n_iter} iterations...")

    with tqdm(total=n_iter) as pb:

        for i in range(n_iter):
            params, opt_state, loss = update_step(params, opt_state)

            #if (i + 1) % print_every == 0:
            losses.append(float(loss))
            pb.set_description(f"loss = {float(loss):.4e}")
            pb.update()

    final_loss = float(loss_fn(params))
    if logger is not None:
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


def infer_labels(flux: np.ndarray,
                 ivar: np.ndarray,
                 theta: np.ndarray,
                 H: np.ndarray,
                 label_mean: np.ndarray,
                 label_std: np.ndarray,
                 scatter: np.ndarray,
                 init_labels_std: np.ndarray = None,
                 n_iter: int = 2000,
                 learning_rate: float = 0.05,
                 seed:int = 42,
                 optimizer: str = 'adam',
                 schedule = None,
                 grid_points: int = 5,
                 grid_range:tuple = (-3.0, 3.0),
                 fixed_mask: np.ndarray = None,
                 fixed_values_phys: np.ndarray = None,
                 logger=None,
                 batch_size_bfgs: int = 100):
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
    schedule
        optional scheudle to pass to adam
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
    batch_size_bfgs: int
         batch size for bfgs. Need to play with for GPU memory 

    Returns:
    --------
    labels : array (n_stars, 4)
        Inferred stellar labels (teff, logg, m_h, alpha_h)
    cov_matrices_phys: array (n_stars, 4, 4)
        full inverse of hessian covariance matrix
    """
    from scipy.optimize import minimize
    from itertools import product

    n_stars, n_wavelengths = flux.shape
    n_labels = len(label_mean)

    if logger is not None:
        logger.info(f"Inferring labels for {n_stars} stars using {optimizer.upper()}...")
    else:
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
        W = jnn.softplus(design_vec @ theta_jnp)
        pred_flux = 1.0 - W @ H_jnp
        total_var = var_single + scatter_sq
        chi_sq = (flux_single - pred_flux)**2 / total_var
        return 0.5 * jnp.sum(chi_sq)

    single_star_loss_and_grad = jit(jax.value_and_grad(single_star_loss))

    # =====================================
    # RUN GRID SEARCH IF NO STARTING VALUES
    # ======================================
    if init_labels_std is None:
        # Build the grid once (shared across all stars)
        if isinstance(grid_range, tuple):
            if logger is not None:
                logger.info(f"  Performing batched grid search ({grid_points}^{n_labels} = {grid_points**n_labels} points per star)...")
            else:
                print(f"  Performing batched grid search ({grid_points}^{n_labels} = {grid_points**n_labels} points per star)...")
            grid_1d = jnp.linspace(grid_range[0], grid_range[1], grid_points)
            grid_points_all = jnp.array(list(product(*[grid_1d]*n_labels)))  # (n_grid, n_labels)
        else:
            if logger is not None:
                logger.info(f"  Performing batched grid search {np.prod(grid_points)} points per star)...")
            else:
                print(f"  Performing batched grid search {np.prod(grid_points)} points per star)...")
            grid_1d = [jnp.linspace(grid_range[i][0], grid_range[i][1], grid_points[i]) for i in range(len(grid_range))]
            grid_points_all = jnp.array(list(product(*grid_1d)))
        n_grid = len(grid_points_all)
        
        if logger is not None:
            logger.info(f"    Grid has {n_grid:,} points")
        else:
            print(f"    Grid has {n_grid:,} points")
        
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
        
        if logger is not None:
            logger.info(f"    Processing {n_stars} stars in {n_batches} batches of ~{batch_size}...")
        else:
            print(f"    Processing {n_stars} stars in {n_batches} batches of ~{batch_size}...")
        
        for i in tqdm(range(n_batches), desc="Grid search batches"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_stars)
            
            # Process this batch
            batch_result = grid_search_batch(
                flux_jnp[start_idx:end_idx],
                var_jnp[start_idx:end_idx]
            )
            init_labels_std[start_idx:end_idx] = np.array(batch_result)
        
        if logger is not None:
            logger.info(f"  Grid search complete.")
        else:
            print(f"  Grid search complete.")

    # =====================================
    # RUN WITH FIXED LABELS FOR EM-STYLE
    # ======================================
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

    # =====================================
    # RUN INFERENCE WITH DESIRED OPTIMIZER
    # ======================================
    if optimizer == 'two-stage' or optimizer == 'adam':
        # do adam and save as init if two-stage warmup
        params = {'labels_std': jnp.array(init_labels_std)}

        @jit
        def forward(params, theta_jnp, H_jnp):
            """Compute predicted flux from labels."""
            labels_std = params['labels_std']
            design_matrix = build_design_matrix_batch_jax(labels_std)
            W = jnn.softplus(design_matrix @ theta_jnp)
            pred_flux = 1.0 - W @ H_jnp
            return pred_flux

        @jit
        def loss_fn(params, theta_jnp, H_jnp, flux_jnp, var_jnp, scatter_sq):
            """Compute weighted reconstruction loss."""
            pred_flux = forward(params, theta_jnp, H_jnp)
            total_var = var_jnp + scatter_sq
            chi_sq = (flux_jnp - pred_flux)**2 / total_var
            return 0.5 * jnp.sum(chi_sq) / (n_stars * n_wavelengths)

        @jit
        def loss_and_grad(params, theta_jnp, H_jnp, flux_jnp, var_jnp, scatter_sq):
            return jax.value_and_grad(loss_fn)(
                params, theta_jnp, H_jnp, flux_jnp, var_jnp, scatter_sq
    )

        if schedule is not None:
            opt = optax.adam(schedule)
        else:
            opt = optax.adam(learning_rate)
        opt_state = opt.init(params)

        @jit
        def update_step(params, opt_state,
                        theta_jnp, H_jnp, flux_jnp, var_jnp, scatter_sq):
            
            loss, grads = loss_and_grad(
                params, theta_jnp, H_jnp, flux_jnp, var_jnp, scatter_sq
            )
            
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            
            return params, opt_state, loss

        if optimizer == 'adam':
            n_iteri = n_iter
        else:
            n_iteri = n_iter[0]
        with tqdm(total=n_iteri) as pb:
            for i in range(n_iteri):
                params, opt_state, loss = update_step(
                    params, opt_state,
                    theta_jnp, H_jnp, flux_jnp, var_jnp, scatter_sq
                )
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
        
        if logger is not None:
            logger.info(f"  Running L-BFGS-B optimization (GPU-accelerated)...")
        else:
            print(f"  Running L-BFGS-B optimization (GPU-accelerated)...")
        labels_std_final = np.zeros((n_stars, n_labels))
        
        # Create solver
        if optimizer == 'bfgs':
            n_iteri = n_iter
        else:
            n_iteri = n_iter[1]
        solver = jaxopt.LBFGS(fun=single_star_loss, maxiter=n_iteri, tol=1e-6)
        
        # Batch size - adjust based on GPU memory (100 is safe for most GPUs)
        n_batches = (n_stars + batch_size_bfgs - 1) // batch_size_bfgs

        # see if multiple devises
        n_devices = jax.device_count()
        print(f'Optimizing on {n_devices} GPUs')
        if n_devices > 1:
            @pmap
            def optimize_batch(init_batch, flux_batch, var_batch):
                def optimize_single(init_single, flux_single, var_single):
                    result = solver.run(init_single, flux_single, var_single)
                    return result.params
                return vmap(optimize_single)(init_batch, flux_batch, var_batch)
        else:
            @jit
            def optimize_batch(init_batch, flux_batch, var_batch):
                def optimize_single(init_single, flux_single, var_single):
                    result = solver.run(init_single, flux_single, var_single)
                    return result.params
                return vmap(optimize_single)(init_batch, flux_batch, var_batch)

        for i in tqdm(range(n_batches), desc="BFGS batches"):
            start_idx = i * batch_size_bfgs
            end_idx = min((i + 1) * batch_size_bfgs, n_stars)

            batch_init = jnp.array(init_labels_std[start_idx:end_idx])
            batch_flux = flux_jnp[start_idx:end_idx]
            batch_var  = var_jnp[start_idx:end_idx]

            actual_size = end_idx - start_idx

            if n_devices > 1:
                # Pad to nearest multiple of n_devices if needed
                remainder = actual_size % n_devices
                if remainder != 0:
                    pad = n_devices - remainder
                    batch_init = jnp.concatenate([batch_init, jnp.zeros((pad, batch_init.shape[-1]))], axis=0)
                    batch_flux = jnp.concatenate([batch_flux, jnp.zeros((pad, batch_flux.shape[-1]))], axis=0)
                    batch_var  = jnp.concatenate([batch_var,  jnp.zeros((pad, batch_var.shape[-1]))],  axis=0)

                per_device = batch_init.shape[0] // n_devices
                result = optimize_batch(
                    batch_init.reshape(n_devices, per_device, -1),
                    batch_flux.reshape(n_devices, per_device, -1),
                    batch_var.reshape(n_devices, per_device, -1)
                ).reshape(-1, batch_init.shape[-1])[:actual_size]  # slice off padding
            else:
                result = optimize_batch(batch_init, batch_flux, batch_var)

            labels_std_final[start_idx:end_idx] = np.array(result)

    if optimizer not in ['adam', 'bfgs', 'two-stage']:
        raise ValueError('Provide invalid optimizer string!')

    # =====================================
    # GET INVERSE OF HESSIAN
    # ======================================
    hessian_fn = jit(vmap(
        jax.hessian(single_star_loss, argnums=0),
        in_axes=(0, 0, 0)
    ))
    inv_fn = jit(vmap(lambda h: jnp.linalg.inv(h + 1e-6 * jnp.eye(n_labels))))

    # Compute in batches to avoid memory issues
    cov_batch_size = 500
    n_batches_cov = (n_stars + cov_batch_size - 1) // cov_batch_size
    cov_matrices_std = np.zeros((n_stars, n_labels, n_labels))

    for i in tqdm(range(n_batches_cov), desc="Computing Hessians"):
        s = i * cov_batch_size
        e = min((i + 1) * cov_batch_size, n_stars)
        H_batch = hessian_fn(
            jnp.array(labels_std_final[s:e]),
            flux_jnp[s:e],
            var_jnp[s:e]
        )
        cov_matrices_std[s:e] = np.array(inv_fn(H_batch))

    # Convert covariance from standardized to physical units
    # If labels_phys = labels_std * label_std + label_mean,
    # then Cov_phys = diag(label_std) @ Cov_std @ diag(label_std)
    scale = label_std[:, None] * label_std[None, :]  # (n_labels, n_labels)
    cov_matrices_phys = cov_matrices_std * scale[None, :, :]  # broadcast over stars

    # Convert back to physical labels
    labels_final = labels_std_final * label_std + label_mean

    return labels_final, cov_matrices_phys
