import unittest

import numpy as np

from activations import relu_derivative


class ReluDerivativeEdgeTests(unittest.TestCase):
    def test_zero_has_zero_slope(self):
        self.assertEqual(relu_derivative(np.array([0.0])).tolist(), [0.0])

    def test_mixed_values(self):
        result = relu_derivative(np.array([-3.0, -0.001, 0.001, 5.0]))
        self.assertEqual(result.tolist(), [0.0, 0.0, 1.0, 1.0])

    def test_output_is_float(self):
        self.assertEqual(relu_derivative(np.array([1.0, -1.0])).dtype, float)


if __name__ == "__main__":
    unittest.main()
