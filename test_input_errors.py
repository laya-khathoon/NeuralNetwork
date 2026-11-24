"""Bad input must raise a clear ValueError, not a confusing crash."""
from evaluation import evaluate_classification
from losses import binary_cross_entropy


def expect_value_error(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except ValueError:
        return
    raise AssertionError("expected a ValueError")


expect_value_error(evaluate_classification, [], [])
expect_value_error(evaluate_classification, [1, 0], [0.5])
expect_value_error(evaluate_classification, [2, 0], [0.5, 0.5])
expect_value_error(evaluate_classification, [1, 0], [1.5, 0.5])
expect_value_error(evaluate_classification, [1, 0], [0.5, float("nan")])
expect_value_error(evaluate_classification, [1, 0], [0.5, 0.5], threshold=1.5)
expect_value_error(binary_cross_entropy, [1, 0], [0.5])
expect_value_error(binary_cross_entropy, [1, 3], [0.5, 0.5])

print("Input error tests passed.")
