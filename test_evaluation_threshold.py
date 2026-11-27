import unittest

from evaluation import evaluate_classification


class EvaluationThresholdTests(unittest.TestCase):
    def test_threshold_is_inclusive(self):
        result = evaluate_classification([1], [0.5], threshold=0.5)
        self.assertEqual(result["true_positive"], 1)

    def test_higher_threshold_turns_positives_into_misses(self):
        y_true, y_pred = [1, 1, 0], [0.6, 0.9, 0.3]
        low = evaluate_classification(y_true, y_pred, threshold=0.5)
        high = evaluate_classification(y_true, y_pred, threshold=0.8)
        self.assertEqual(low["false_negative"], 0)
        self.assertEqual(high["false_negative"], 1)

    def test_threshold_one_predicts_only_certain_positives(self):
        result = evaluate_classification([1, 0], [1.0, 0.99], threshold=1.0)
        self.assertEqual((result["true_positive"], result["false_positive"]), (1, 0))


if __name__ == "__main__":
    unittest.main()
