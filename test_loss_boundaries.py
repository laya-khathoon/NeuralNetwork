"""Loss endpoint clipping, known values and invalid-label behavior."""
import unittest

import numpy as np

from losses import binary_cross_entropy


class LossBoundaryTests(unittest.TestCase):
    def test_perfect_endpoints_are_finite_and_near_zero(self):
        with np.errstate(divide="raise", invalid="raise"):
            loss = binary_cross_entropy([0, 1], [0.0, 1.0])
        self.assertTrue(np.isfinite(loss))
        self.assertLess(loss, 1e-9)

    def test_incorrect_endpoints_have_large_finite_loss(self):
        loss = binary_cross_entropy([1, 0], [0.0, 1.0])
        self.assertTrue(np.isfinite(loss))
        self.assertGreater(loss, 20)

    def test_uncertain_predictions_equal_log_two(self):
        self.assertAlmostEqual(binary_cross_entropy([0, 1], [0.5, 0.5]),
                               float(np.log(2)))

    def test_rejects_shape_mismatch_instead_of_broadcasting(self):
        with self.assertRaisesRegex(ValueError, "same shape"):
            binary_cross_entropy([[0], [1]], [0.2, 0.8])

    def test_rejects_non_binary_labels(self):
        for labels in ([0, 2], [0.2, 1], [-1, 1], [float("nan"), 1]):
            with self.subTest(labels=labels):
                with self.assertRaisesRegex(ValueError, "0 or 1"):
                    binary_cross_entropy(labels, [0.2, 0.8])


if __name__ == "__main__":
    unittest.main()
