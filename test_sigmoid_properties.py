"""Property checks for sigmoid and relu."""
import numpy as np

from activations import relu, sigmoid, sigmoid_derivative

x = np.linspace(-10, 10, 41)

# Sigmoid stays strictly inside (0, 1) and is symmetric: s(-x) = 1 - s(x).
assert np.all((sigmoid(x) > 0) & (sigmoid(x) < 1))
assert np.allclose(sigmoid(-x), 1 - sigmoid(x))

# Sigmoid always grows when the input grows.
assert np.all(np.diff(sigmoid(x)) > 0)

# The derivative peaks at 0.25, where the sigmoid output is 0.5.
assert np.isclose(sigmoid_derivative(sigmoid(np.array([0.0])))[0], 0.25)

# ReLU never returns a negative number.
assert np.all(relu(x) >= 0)

print("Sigmoid property tests passed.")
