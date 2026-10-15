
import numpy as np

from activations import relu, relu_derivative, sigmoid
from layers import DenseLayer
from losses import binary_cross_entropy


class NeuralNetwork:
    """A two-layer neural network for binary classification."""

    def __init__(self, input_size, hidden_size, output_size=1, seed=42):
        if output_size != 1:
            raise ValueError("This network supports one output neuron.")

        self.hidden_layer = DenseLayer(
            input_size, hidden_size, seed=seed
        )
        self.output_layer = DenseLayer(
            hidden_size, output_size, seed=seed + 1
        )

        self.hidden_z = None
        self.predictions = None
        self.loss_history = []

    def forward(self, inputs):
        """Perform forward propagation."""
        self.hidden_z = self.hidden_layer.forward(inputs)
        hidden_output = relu(self.hidden_z)

        output_z = self.output_layer.forward(hidden_output)
        self.predictions = sigmoid(output_z)

        return self.predictions

    def backward(self, y_true):
        """Perform backpropagation."""
        y_true = np.asarray(y_true, dtype=float).reshape(-1, 1)

        if self.predictions is None:
            raise ValueError("Run forward() before backward().")

        if y_true.shape != self.predictions.shape:
            raise ValueError(
                "Labels and predictions must have matching shapes."
            )

        doutput = self.predictions - y_true
        dhidden = self.output_layer.backward(doutput)
        dhidden *= relu_derivative(self.hidden_z)

        self.hidden_layer.backward(dhidden)

    def update_parameters(self, learning_rate):
        """Update weights and biases."""
        self.hidden_layer.update_parameters(learning_rate)
        self.output_layer.update_parameters(learning_rate)

    def train_step(self, inputs, y_true, learning_rate=0.1):
        """Perform one training step and return its loss."""
        predictions = self.forward(inputs)
        loss = binary_cross_entropy(y_true, predictions)

        self.backward(y_true)
        self.update_parameters(learning_rate)

        return loss

    def fit(self, inputs, y_true, epochs=1000,
            learning_rate=0.1, verbose=True):
        """Train the network and record loss at every epoch."""
        if epochs < 1:
            raise ValueError("Epochs must be at least 1.")

        self.loss_history = []

        for epoch in range(epochs):
            loss = self.train_step(
                inputs,
                y_true,
                learning_rate=learning_rate
            )

            self.loss_history.append(loss)

            if verbose and (epoch + 1) % 500 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs} - "
                    f"Loss: {loss:.6f}"
                )

        return self.loss_history
