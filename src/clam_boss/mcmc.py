import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import numpy as np
import jax
import jax.numpy as jnp
import jax.nn as jnn
from jax import jit

from clam_boss.model import build_design_matrix_batch_jax


@jit
def forward(labels_std, theta_jnp, H_jnp):
    """Compute predicted flux from labels."""
    design_matrix = build_design_matrix_batch_jax(labels_std)
    W = jnn.softplus(design_matrix @ theta_jnp)
    pred_flux = 1.0 - W @ H_jnp
    return pred_flux


def mcmc_params(flux: np.ndarray,
                ivar: np.ndarray,
                best_labels: np.ndarray,
                theta: np.ndarray,
                H: np.ndarray,
                label_mean: np.ndarray,
                label_std: np.ndarray,
                scatter: np.ndarray,
                label_sigma: np.ndarray,
                num_warmup: int = 750,
                num_samples: int = 300,
                num_chains: int = 4,
                seed: int = 0):
    """
    Run MCMC to get posterior estimates from the best fit parameters
    of the inferred forward model.

    Parameters:
    -----------
    flux : array (n_stars, n_wavelengths)
        Normalized flux spectra
    ivar : array (n_stars, n_wavelengths)
        Inverse variance weights
    best_labels : array (n_stars, n_labels)
        The best fit labels from the inference stage. In physical units
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
    label_sigma : array (n_labels,) or (n_stars, n_labels)
        Prior std (physical units) around best_labels for each label
        dimension, e.g. from the inverse-Hessian covariance returned by
        infer_labels. Broadcasts against best_labels.
    seed : int
        PRNG seed for the MCMC run
    """
    n_stars, n_wavelengths = flux.shape
    n_labels = len(label_mean)
    n_samples = int(num_samples * num_chains)

    # Convert to JAX arrays
    flux_jnp = jnp.array(flux)
    var_jnp = 1.0 / jnp.maximum(ivar, 1e-16)
    theta_jnp = jnp.array(theta)
    H_jnp = jnp.array(H)
    scatter_sq = jnp.array(scatter)**2
    label_mean_jnp = jnp.array(label_mean)
    label_std_jnp = jnp.array(label_std)
    best_labels_jnp = jnp.array(best_labels)
    label_sigma_jnp = jnp.broadcast_to(jnp.array(label_sigma), best_labels_jnp.shape)

    def model(flux_jnp, var_jnp, theta_jnp, H_jnp, scatter_sq, best_labels,
              label_sigma, label_mean, label_std):
        """
        MCMC model, jointly over all stars.
        """
        with numpyro.plate('stars', n_stars):
            teff = numpyro.sample('teff', dist.Normal(best_labels[:, 0], label_sigma[:, 0]))
            logg = numpyro.sample('logg', dist.Normal(best_labels[:, 1], label_sigma[:, 1]))
            fe_h = numpyro.sample('fe_h', dist.Normal(best_labels[:, 2], label_sigma[:, 2]))
            alpha_m = numpyro.sample('alpha_m', dist.Normal(best_labels[:, 3], label_sigma[:, 3]))

        labels = jnp.stack([teff, logg, fe_h, alpha_m], axis=-1)  # (n_stars, n_labels)
        labels_std = (labels - label_mean) / label_std

        pred_flux = forward(labels_std, theta_jnp, H_jnp)
        total_var = var_jnp + scatter_sq
        chi_sq = (flux_jnp - pred_flux)**2 / total_var
        numpyro.factor("loglike", -0.5 * jnp.sum(chi_sq))

    init_strategy = numpyro.infer.init_to_value(values={
        'teff': best_labels_jnp[:, 0],
        'logg': best_labels_jnp[:, 1],
        'fe_h': best_labels_jnp[:, 2],
        'alpha_m': best_labels_jnp[:, 3],
    })
    nuts_kernel = NUTS(model, init_strategy=init_strategy)
    mcmc = MCMC(nuts_kernel,
                num_warmup=num_warmup,
                num_samples=num_samples,
                num_chains=num_chains,
                chain_method='vectorized')
    mcmc.run(jax.random.PRNGKey(seed),
             flux_jnp, var_jnp,
             theta_jnp, H_jnp, scatter_sq,
             best_labels_jnp, label_sigma_jnp,
             label_mean_jnp, label_std_jnp)
    samples = mcmc.get_samples()  # each site: (num_chains * num_samples, n_stars)

    mcmc.print_summary()

    labels_samples = np.zeros((n_stars, n_labels, n_samples))
    labels_samples[:, 0, :] = np.array(samples['teff']).T
    labels_samples[:, 1, :] = np.array(samples['logg']).T
    labels_samples[:, 2, :] = np.array(samples['fe_h']).T
    labels_samples[:, 3, :] = np.array(samples['alpha_m']).T

    return labels_samples
