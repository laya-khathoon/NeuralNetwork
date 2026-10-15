import numpy as np


class DenseLayer:
    """A fully connected neural network layer."""

    def __init__(self, input_size, output_size, seed=None):
        rng = np.random.default_rng(seed)

        self.weights = (
            rng.standard_normal((input_size, output_size))
            * np.sqrt(2.0 / input_size)
        )
        self.biases = np.zeros((1, output_size))

        self.inputs = None
        self.dweights = None
        self.dbiases = None

    def forward(self, inputs):
        """Calculate the layer output."""
        self.inputs = np.asarray(inputs, dtype=float)
        return self.inputs @ self.weights + self.biases

    def backward(self, dvalues):
        """Calculate gradients for backpropagation."""
        samples = self.inputs.shape[0]

        self.dweights = self.inputs.T @ dvalues / samples
        self.dbiases = np.sum(
            dvalues, axis=0, keepdims=True
        ) / samples

        dinputs = dvalues @ self.weights.T
        return dinputs

    def update_parameters(self, learning_rate):
        """Update weights and biases using gradient descent."""
        self.weights -= learning_rate * self.dweights
        self.biases -= learning_rate * self.dbiases