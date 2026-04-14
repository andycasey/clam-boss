import numpy as np
import json
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline


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


def save_MLP_npz(model, save_path='nmf_MLP_model.npz'):
    """
    Save the resulting MLP. weights go in npz, metdaata in json
    """
    x_scaler = model.named_steps["x_scaler"]
    reg = model.named_steps["reg"]
    mlp = reg.regressor_
    y_scaler = reg.transformer_

    # metadata in JSON
    meta = {
        "mlp": {
            "hidden_layer_sizes": mlp.hidden_layer_sizes,
            "activation": mlp.activation,
            "n_layers": len(mlp.coefs_),
            "input_dim": mlp.coefs_[0].shape[0],
            "output_dim": mlp.coefs_[-1].shape[1]
        },
        "shapes": {
            "coefs": [w.shape for w in mlp.coefs_],
            "intercepts": [b.shape for b in mlp.intercepts_]
        }
    }

    with open(save_path.replace('.npz', '.json'), "w") as f:
        json.dump(meta, f)

    # save weights in NPZ
    arrays = {}

    # X scaler
    arrays["x_mean"] = x_scaler.mean_
    arrays["x_scale"] = x_scaler.scale_

    # Y scaler
    arrays["y_mean"] = y_scaler.mean_
    arrays["y_scale"] = y_scaler.scale_

    # MLP weights
    for i, (W, b) in enumerate(zip(mlp.coefs_, mlp.intercepts_)):
        arrays[f"W_{i}"] = W
        arrays[f"b_{i}"] = b

    np.savez_compressed(save_path, **arrays)
    return


def train_and_save_MLP(W_train, train_labels,
                       save_path='nmf_MLP_model.npz',
                       hidden_layer_sizes=(512, 256, 128, 64),
                       activation="tanh",
                       alpha=1e-5,
                       max_iter=1000,
                       n_iter_no_change=50,
                       logger=None):
    """
    Train a basic MLP to predict stellar labels from
    NMF weights
    """
    mlp = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        alpha=alpha,
        max_iter=max_iter,
        n_iter_no_change=n_iter_no_change,
        early_stopping=True,
        random_state=42
    )

    model = Pipeline([
        ("x_scaler", StandardScaler()),
        ("reg", TransformedTargetRegressor(
            regressor=mlp,
            transformer=StandardScaler()
        ))
    ])

    model.fit(W_train, train_labels)

    save_MLP_npz(model, save_path=save_path)

    if logger is not None:
        logger.info(f"\nModel saved to {save_path}")
    
    return


# Activiations for MLP
def relu(x): return np.maximum(0, x)
def tanh(x): return np.tanh(x)
def identity(x): return x
def logistic(x): return 1 / (1 + np.exp(-x))

ACTIVATIONS = {
    "relu": relu,
    "tanh": tanh,
    "identity": identity,
    "logistic": logistic
}


class StandardScalerManual:
    """
    Manual scalar for the MLP
    """
    def __init__(self, mean, scale):
        self.mean = mean
        self.scale = scale

    def transform(self, X):
        return (X - self.mean) / self.scale

    def inverse_transform(self, X):
        return X * self.scale + self.mean


class MLPManual:
    """
    Recreate the MLP from just the weights
    """
    def __init__(self, weights, biases, activation):
        self.weights = weights
        self.biases = biases
        self.activation = ACTIVATIONS[activation]

    def predict(self, X):
        out = X
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            out = out @ W + b
            if i < len(self.weights) - 1:
                out = self.activation(out)
        return out


class ManualPipeline:
    """
    Full manual pipeline
    """
    def __init__(self, meta, arrays):
        self.expected_input_dim = meta["mlp"]["input_dim"]
        self.expected_output_dim = meta["mlp"]["output_dim"]
        # scalers
        self.x_scaler = StandardScalerManual(
            arrays["x_mean"], arrays["x_scale"]
        )
        self.y_scaler = StandardScalerManual(
            arrays["y_mean"], arrays["y_scale"]
        )

        # weights
        n_layers = meta["mlp"]["n_layers"]
        weights = [arrays[f"W_{i}"] for i in range(n_layers)]
        biases = [arrays[f"b_{i}"] for i in range(n_layers)]

        self.mlp = MLPManual(
            weights,
            biases,
            meta["mlp"]["activation"]
        )

    def predict(self, X):
        # make sure shape matches
        if X.shape[1] != self.expected_input_dim:
            raise ValueError(
                f"Input feature dimension mismatch: "
                f"expected {self.expected_input_dim}, got {X.shape[1]}"
            )

        Xs = self.x_scaler.transform(X)
        ys = self.mlp.predict(Xs)

        # check output shape
        if ys.shape[1] != self.expected_output_dim:
            raise ValueError(
                f"Output dimension mismatch: "
                f"expected {self.expected_output_dim}, got {ys.shape[1]}"
            )

        return self.y_scaler.inverse_transform(ys)


def load_MLP_model(save_path='nmf_MLP_model.npz'):
    """
    Load the MLP meta data and weights to do predictions
    """
    # load metadata
    with open(save_path.replace('.npz', '.json'), "r") as f:
        meta = json.load(f)

    # load arrays
    arrays_npz = np.load(save_path)
    arrays = {k: arrays_npz[k] for k in arrays_npz.files}

    return ManualPipeline(meta, arrays)

