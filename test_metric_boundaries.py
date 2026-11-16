"""Confusion counts, zero denominators and prediction validation."""
import unittest

from evaluation import evaluate_classification


class MetricBoundaryTests(unittest.TestCase):
    def test_mixed_confusion_matrix_and_scores(self):
        result = evaluate_classification([0, 0, 1, 1], [0.1, 0.8, 0.2, 0.9])
        for key in ("true_positive", "true_negative", "false_positive", "false_negative"):
            self.assertEqual(result[key], 1)
        for key in ("accuracy", "precision", "recall", "f1_score"):
            self.assertEqual(result[key], 0.5)

    def test_no_positive_predictions_has_defined_scores(self):
        result = evaluate_classification([0, 0], [0.1, 0.2])
        self.assertEqual(result["accuracy"], 1.0)
        for key in ("precision", "recall", "f1_score"):
            self.assertEqual(result[key], 0.0)

    def test_probability_equal_to_threshold_is_positive(self):
        result = evaluate_classification([1], [0.7], threshold=0.7)
        self.assertEqual(result["true_positive"], 1)

    def test_invalid_probabilities_are_rejected(self):
        for value in (-0.1, 1.1, float("nan"), float("inf")):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    evaluate_classification([1], [value])

    def test_empty_mismatched_and_invalid_threshold_inputs(self):
        for labels, predictions, threshold in (([], [], 0.5),
                ([0, 1], [0.2], 0.5), ([1], [0.8], -0.1),
                ([1], [0.8], 1.1)):
            with self.subTest(labels=labels, predictions=predictions, threshold=threshold):
                with self.assertRaises(ValueError):
                    evaluate_classification(labels, predictions, threshold)


if __name__ == "__main__":
    unittest.main()
