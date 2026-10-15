
import numpy as np

from activations import (
    sigmoid,
    relu,
    sigmoid_derivative,
    relu_derivative,
)

values = np.array([-2.0, 0.0, 2.0])

print("Input:", values)
print("Sigmoid:", sigmoid(values))
print("ReLU:", relu(values))
print("Sigmoid derivative:", sigmoid_derivative(sigmoid(values)))
print("ReLU derivative:", relu_derivative(values))

assert np.allclose(relu(values), [0.0, 0.0, 2.0])
assert np.allclose(sigmoid(values), [0.11920292, 0.5, 0.88079708])
assert np.allclose(relu_derivative(values), [0.0, 0.0, 1.0])

print("\nAll activation function tests passed.")
