"""Extreme activation inputs and analytical derivative checks."""
import unittest

import numpy as np

from activations import relu, relu_derivative, sigmoid, sigmoid_derivative


class ActivationStabilityTests(unittest.TestCase):
    def test_extreme_sigmoid_inputs_do_not_overflow(self):
        with np.errstate(over="raise", invalid="raise"):
            result = sigmoid(np.array([-1000.0, 0.0, 1000.0]))
        np.testing.assert_allclose(result, [0.0, 0.5, 1.0])

    def test_sigmoid_preserves_matrix_shape_and_bounds(self):
        values = np.arange(-6, 6).reshape(3, 4)
        result = sigmoid(values)
        self.assertEqual(result.shape, values.shape)
        self.assertTrue(np.all((result >= 0) & (result <= 1)))
        self.assertTrue(np.all(np.diff(result.reshape(-1)) > 0))

    def test_sigmoid_derivative_matches_finite_difference(self):
        values = np.array([-2.0, 0.0, 2.0])
        step = 1e-5
        numerical = (sigmoid(values + step) - sigmoid(values - step)) / (2 * step)
        np.testing.assert_allclose(sigmoid_derivative(sigmoid(values)), numerical,
                                   rtol=1e-5, atol=1e-8)

    def test_relu_at_negative_zero_and_positive_inputs(self):
        values = np.array([[-2.0, 0.0, 3.0]])
        np.testing.assert_array_equal(relu(values), [[0.0, 0.0, 3.0]])
        np.testing.assert_array_equal(relu_derivative(values), [[0.0, 0.0, 1.0]])


if __name__ == "__main__":
    unittest.main()
