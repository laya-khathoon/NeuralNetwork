import numpy as np


def binary_cross_entropy(y_true, y_pred):
    """Calculate binary cross-entropy loss."""

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape."
        )

    if not np.all((y_true == 0) | (y_true == 1)):
        raise ValueError("Labels must contain only 0 or 1.")

    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)

    loss = -np.mean(
        y_true * np.log(y_pred)
        + (1.0 - y_true) * np.log(1.0 - y_pred)
    )

    return float(loss)