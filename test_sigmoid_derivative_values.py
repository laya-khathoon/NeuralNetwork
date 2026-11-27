import unittest

import numpy as np

from activations import sigmoid, sigmoid_derivative


class SigmoidDerivativeValueTests(unittest.TestCase):
    def test_peak_is_a_quarter_at_half(self):
        self.assertAlmostEqual(float(sigmoid_derivative(np.array([0.5]))[0]), 0.25)

    def test_zero_at_saturated_outputs(self):
        result = sigmoid_derivative(np.array([0.0, 1.0]))
        self.assertEqual(result.tolist(), [0.0, 0.0])

    def test_matches_numerical_slope(self):
        x, h = 0.7, 1e-6
        numeric = (sigmoid(np.array([x + h])) - sigmoid(np.array([x - h]))) / (2 * h)
        analytic = sigmoid_derivative(sigmoid(np.array([x])))
        self.assertTrue(np.allclose(numeric, analytic, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
