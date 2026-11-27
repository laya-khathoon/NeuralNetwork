import unittest

import numpy as np

from predictor import predict


class FakeNetwork:
    def __init__(self, outputs):
        self.outputs = np.array(outputs, dtype=float).reshape(-1, 1)

    def forward(self, X):
        return self.outputs


class PredictorThresholdTests(unittest.TestCase):
    def test_probability_equal_to_threshold_is_positive(self):
        _, classes = predict(FakeNetwork([0.5]), [[1, 2]], threshold=0.5)
        self.assertEqual(classes.tolist(), [[1]])

    def test_threshold_zero_makes_everything_positive(self):
        _, classes = predict(FakeNetwork([0.0, 0.2]), [[1], [2]], threshold=0.0)
        self.assertEqual(classes.ravel().tolist(), [1, 1])

    def test_threshold_outside_range_is_refused(self):
        with self.assertRaises(ValueError):
            predict(FakeNetwork([0.5]), [[1]], threshold=1.5)


if __name__ == "__main__":
    unittest.main()
