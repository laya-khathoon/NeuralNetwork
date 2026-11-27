import unittest

from losses import binary_cross_entropy


class LossSymmetryTests(unittest.TestCase):
    def test_perfect_prediction_is_near_zero(self):
        self.assertLess(binary_cross_entropy([1, 0], [1.0, 0.0]), 1e-9)

    def test_confident_wrong_prediction_is_large(self):
        self.assertGreater(binary_cross_entropy([1, 0], [0.0, 1.0]), 20)

    def test_even_guess_equals_log_two(self):
        self.assertAlmostEqual(binary_cross_entropy([1, 0], [0.5, 0.5]), 0.6931471805599453)

    def test_flipping_labels_and_predictions_gives_same_loss(self):
        a = binary_cross_entropy([1, 0, 1], [0.9, 0.2, 0.6])
        b = binary_cross_entropy([0, 1, 0], [0.1, 0.8, 0.4])
        self.assertAlmostEqual(a, b)


if __name__ == "__main__":
    unittest.main()
