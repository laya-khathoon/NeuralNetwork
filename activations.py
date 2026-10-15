
import numpy as np


def sigmoid(x):
    """Convert input values to a range between 0 and 1."""
    x = np.asarray(x, dtype=float)

    # Stable calculation to avoid numerical overflow
    result = np.empty_like(x)
    positive = x >= 0

    result[positive] = 1.0 / (1.0 + np.exp(-x[positive]))

    exp_x = np.exp(x[~positive])
    result[~positive] = exp_x / (1.0 + exp_x)

    return result


def relu(x):
    """Return zero for negative values and x for positive values."""
    return np.maximum(0, x)


def sigmoid_derivative(x):
    """Calculate the sigmoid derivative from its output."""
    return x * (1.0 - x)


def relu_derivative(x):
    """Calculate the ReLU derivative."""
    return (x > 0).astype(float)
