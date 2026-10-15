import numpy as np

from activations import relu, relu_derivative, sigmoid
from layers import DenseLayer
from losses import binary_cross_entropy


class NeuralNetwork:
    """A two-layer neural network for binary classification."""

    def __init__(self, input_size, hidden_size, output_size=1, seed=42):
        if output_size != 1:
            raise ValueError(
                "This network supports one output neuron."
            )

        self.hidden_layer = DenseLayer(
            input_size, hidden_size, seed=seed
        )

        self.output_layer = DenseLayer(
            hidden_size, output_size, seed=seed + 1
        )

        self.hidden_z = None
        self.predictions = None

    def forward(self, inputs):
        """Perform forward propagation."""

        hidden_z = self.hidden_layer.forward(inputs)
        self.hidden_z = hidden_z

        hidden_output = relu(hidden_z)

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

        # Do not divide by the sample count here.
        # DenseLayer.backward() averages parameter gradients.
        doutput = self.predictions - y_true

        dhidden = self.output_layer.backward(doutput)

        dhidden *= relu_derivative(self.hidden_z)

        self.hidden_layer.backward(dhidden)

    def update_parameters(self, learning_rate):
        """Update parameters in both layers."""

        self.hidden_layer.update_parameters(learning_rate)
        self.output_layer.update_parameters(learning_rate)

    def train_step(self, inputs, y_true, learning_rate=0.1):
        """Run one training step and return the loss."""

        predictions = self.forward(inputs)

        loss = binary_cross_entropy(y_true, predictions)

        self.backward(y_true)
        self.update_parameters(learning_rate)

        return loss