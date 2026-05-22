# BOSS-CLAM

The CLAM (Constrained Linear Absorption Model) is a constrained linear absorption model that simultaneously fits stellar absorption and continuum via non-negative matrix factorization [(Casey et al. 2026)](https://ui.adsabs.harvard.edu/abs/2026ApJ...998..192C/abstract). This repository is for the implementation of a similar method dubbed "BOSS-CLAM". BOSS-CLAM seeks to decompose stellar spectra using non-negative matrix factorization (NMF). In this method though, we will optimize a mapping between the NMF weights and stellar labels. Once the model is trained, this allows us to infer stellar labels from observed spectra.

This repository is for the BOSS-CLAM model trained on SDSS-V BOSS spectra from DR20 (Medan et al.). This repository hosts the code to train a BOSS-CLAM model, the trained model on DR20 data, and some validation plots using open and globular clusters.

## Installation

To install the BOSS-CLAM code to either train your own model or access the model trained on SDSS-V DR20 data is best done with [uv](https://docs.astral.sh/uv/):
```
git clone https://github.com/andycasey/clam-boss
cd clam-boss
uv venv --python 3.11
source .venv/bin/activate
uv sync
```
The code is built on [JAX](https://docs.jax.dev/en/latest/). If you want to run the code on a GPU, then instead sync the repository with: `uv sync --extra gpu`.

## Example -- Loading and Using the Trained DR20 Model

This repository can be used to load the trained DR20 model and either fit infer parameters from your own BOSS spectra or forward model spectra from arbitrary stellar parameters. The DR20 model is loaded like:
```python
import numpy as np

import jax.numpy as jnp
from jax import jit
import jax.nn as jnn

import optax

from clam_boss.model import (
    compute_nmf_weights,
    infer_labels,
    build_design_matrix_batch_jax
)

from clam_boss.init_model import load_MLP_model

base_dir = 'model_results/nmf_joint_results_with_scatter_K32_alpha_m_w_wide_binaries_w_MS_w_HS'
res = np.load(f'{base_dir}/joint_model_results.npz')
theta = res['theta']
H = res['H']
label_mean = res['label_mean']
label_std = res['label_std']
scatter = res['scatter']

theta_jnp = jnp.array(theta)
H_jnp = jnp.array(H)
scatter_sq = jnp.array(scatter)**2
```

### Forward Model Spectra

The above model can then be used to forward model spectra of arbitrary stellar parameters. This is done like:
```python
@jit
def spectra_predict(labels_std, theta_jnp, H_jnp, scatter_sq):
    design_matrix = build_design_matrix_batch_jax(labels_std)
    W = jnn.softplus(design_matrix @ theta_jnp)
    pred_flux = 1.0 - W @ H_jnp
    return pred_flux

fe_h = np.linspace(-1., 0., 500)
teff = np.zeros(len(fe_h)) + 5772
logg = np.zeros(len(fe_h)) + 4.44
alpha_m = np.zeros(len(fe_h)) + 0


labels_std = (np.column_stack((teff, logg, fe_h, alpha_m)) - label_mean) / label_std

pred_flux = spectra_predict(labels_std, theta_jnp, H_jnp, scatter_sq)
```

### Infer Parameters for BOSS Spectra

The model can also be used to infer the stellar parameters of BOSS spectra. This assumes that you have three arrays, `flux`, `ivar` and `continuum` of shape `[N, 4648]`, where `N` is the number of stars. Spectra should be resampled to the stellar rest frame.
```python
# normalize spectrum
norm_flux = flux / continuum

# Compute inverse variance for normalized flux
norm_ivar = continuum**2 * ivar

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

# load MLP for initial guess
model_path = f'{base_dir}/nmf_MLP_model.npz'
mlp_model = load_MLP_model(save_path=model_path)

W_vals = compute_nmf_weights(norm_flux, H)
init_labels = mlp_model.predict(W_vals)
init_labels_std = (init_labels - label_mean) / label_std

# Infer labels using adam with decay schedule
inferred_labels, label_covariances = infer_labels(
    norm_flux, norm_ivar,
    theta, H, label_mean, label_std, scatter,
    init_labels_std=init_labels_std,
    n_iter=[500, 100],
    learning_rate=0.01,
    schedule=optax.cosine_decay_schedule(init_value=0.01, decay_steps=500),
    optimizer='two-stage',
    grid_points=None,
    grid_range=None,
    batch_size_bfgs=len(norm_flux)
)
```

## Example -- Training a Model

The code can also be used to train your own model. This example assumes that you have four arrays: `flux`, `ivar` and `continuum` of shape `[N, M]`, where `N` is the number of stars and `M` is the length of the wavelength axis, and `true_labels` of shape `[N, L]`, where `L` is the number of stellar labels per star. Spectra should be resampled to the stellar rest frame.
```python
import numpy as np
from clam_boss.model import joint_optimization

# normalize spectrum
norm_flux = flux / continuum

# Compute inverse variance for normalized flux
norm_ivar = continuum**2 * ivar

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

# assume weights equal for all labels
per_label_weights = np.zeros_like(true_labels) + 1.

# train the model

K = 160  # number of basis vectors for NMF
n_iter = 10_000  # number of iterations for optimization
learning_rate = 0.001  # learning rate for adam
print_every = 1000  # how often to print loss in optimization

inferred_labels, theta, H, W, label_mean, label_std, losses, scatter = joint_optimization(
        norm_flux, norm_ivar, true_labels, K,
        n_iter=n_iter,
        learning_rate=learning_rate,
        print_every=print_every,
        seed=42,
        per_label_weights=per_label_weights
    )

# Save results
np.savez_compressed(f'joint_model_results.npz',
                    inferred_labels=inferred_labels,
                    true_labels=true_labels,
                    theta=theta,
                    H=H,
                    W=W,
                    label_mean=label_mean,
                    label_std=label_std,
                    losses=losses,
                    scatter=scatter)
```

## Scripts and Plots Included

The respiratory also includes the scripts and validation plots for the SDSS-V DR20 BOSS-CLAM value added catalog (Medan et al.):
- [`nmf_stellar_model_joint.py`](https://github.com/andycasey/clam-boss/blob/main/nmf_stellar_model_joint.py): The main script used to train the DR20 model.
- [`model_results/nmf_joint_results_with_scatter_K32_alpha_m_w_wide_binaries_w_MS_w_HS/`](https://github.com/andycasey/clam-boss/tree/main/model_results/nmf_joint_results_with_scatter_K32_alpha_m_w_wide_binaries_w_MS_w_HS): The directory that includes the trained DR20 model and some results plots from the training.
- [`predict_clusters.py`](https://github.com/andycasey/clam-boss/blob/main/predict_clusters.py): Script to infer parameters of open and globular clusters for validating the model.
- [`validation_results/`](https://github.com/andycasey/clam-boss/tree/main/validation_results): Validation plots for the open and globular clusters.