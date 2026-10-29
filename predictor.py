import numpy as np


def predict(network, X, threshold=0.5):
    """
    Predict probabilities and binary classes using a trained network.
    """
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")

    X = np.asarray(X, dtype=float)

    if X.ndim == 1:
        X = X.reshape(1, -1)

    if X.ndim != 2:
        raise ValueError("X must be a 1D or 2D array.")

    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X must not be empty.")

    probabilities = np.asarray(network.forward(X))

    if probabilities.ndim == 1:
        probabilities = probabilities.reshape(-1, 1)

    if probabilities.shape != (X.shape[0], 1):
        raise ValueError(
            "The network must return one probability per input sample."
        )

    if not np.all(np.isfinite(probabilities)):
        raise ValueError("Network predictions must be finite.")

    if np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("Network outputs must be probabilities in [0, 1].")

    predicted_classes = (probabilities >= threshold).astype(int)

    return probabilities, predicted_classes
