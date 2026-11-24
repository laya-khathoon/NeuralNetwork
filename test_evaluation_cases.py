"""Worked examples for evaluate_classification."""
import numpy as np

from evaluation import evaluate_classification

# Perfect predictions.
perfect = evaluate_classification([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2])
assert perfect["accuracy"] == 1.0 and perfect["f1_score"] == 1.0

# Everything predicted negative: no true positives, so precision/recall/F1 are 0.
none_found = evaluate_classification([1, 1, 0, 0], [0.1, 0.2, 0.1, 0.2])
assert none_found["precision"] == 0.0 and none_found["recall"] == 0.0 and none_found["f1_score"] == 0.0
assert none_found["accuracy"] == 0.5

# The threshold changes the class: 0.6 is positive at 0.5 but negative at 0.7.
assert evaluate_classification([1], [0.6], threshold=0.5)["true_positive"] == 1
assert evaluate_classification([1], [0.6], threshold=0.7)["false_negative"] == 1

# Mixed example worked out by hand: tp=2, fp=1, fn=1, tn=1.
mixed = evaluate_classification([1, 1, 1, 0, 0], [0.9, 0.8, 0.2, 0.7, 0.1])
assert (mixed["true_positive"], mixed["false_positive"], mixed["false_negative"], mixed["true_negative"]) == (2, 1, 1, 1)
assert np.isclose(mixed["precision"], 2 / 3) and np.isclose(mixed["recall"], 2 / 3)

print("Evaluation case tests passed.")
