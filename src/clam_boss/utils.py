import os
import numpy as np
import warnings
from tqdm import tqdm, trange
import logging


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
