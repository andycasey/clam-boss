import numpy as np


def save_ridge_model_npz(ridge_model, W_scaler, label_scaler, 
                         save_path='nmf_ridge_model.npz',
                         logger=None):
    """
    Save Ridge model as numpy arrays in NPZ format.
    """
    if logger is not None:
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


def train_and_save_ridge_model(W_train, train_labels,
                               save_path='nmf_ridge_model.npz',
                               alpha=1.0, logger=None):
    """
    Train and save ridge model used for initial guess
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    
    if logger is not None:
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
    save_ridge_model_npz(ridge, W_scaler, label_scaler, save_path, logger=logger)
    
    if logger is not None:
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
