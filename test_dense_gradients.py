"""Dense-layer backward values checked independently against hand calculations."""
import unittest

import numpy as np

from layers import DenseLayer


class DenseGradientTests(unittest.TestCase):
    def test_backward_averages_parameter_gradients_over_batch(self):
        layer = DenseLayer(2, 1, seed=1)
        layer.weights = np.array([[2.0], [3.0]])
        inputs = np.array([[1.0, 2.0], [3.0, 4.0]])
        layer.forward(inputs)
        dinputs = layer.backward(np.array([[1.0], [2.0]]))
        np.testing.assert_allclose(layer.dweights, [[3.5], [5.0]])
        np.testing.assert_allclose(layer.dbiases, [[1.5]])
        np.testing.assert_allclose(dinputs, [[2.0, 3.0], [4.0, 6.0]])

    def test_update_uses_learning_rate_and_stored_gradients(self):
        layer = DenseLayer(2, 1, seed=2)
        layer.forward([[1.0, 2.0]])
        layer.backward(np.array([[0.5]]))
        weights = layer.weights.copy()
        biases = layer.biases.copy()
        layer.update_parameters(0.1)
        np.testing.assert_allclose(layer.weights, weights - 0.1 * layer.dweights)
        np.testing.assert_allclose(layer.biases, biases - 0.1 * layer.dbiases)

    def test_seed_reproduces_initialization(self):
        first, second = DenseLayer(3, 4, seed=42), DenseLayer(3, 4, seed=42)
        np.testing.assert_array_equal(first.weights, second.weights)
        np.testing.assert_array_equal(first.biases, np.zeros((1, 4)))
        self.assertEqual(first.forward(np.ones((5, 3))).shape, (5, 4))


if __name__ == "__main__":
    unittest.main()
