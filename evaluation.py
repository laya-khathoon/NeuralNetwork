
import numpy as np


def evaluate_classification(y_true, y_pred, threshold=0.5):
    """Calculate binary classification evaluation metrics."""

    y_true = np.asarray(y_true).reshape(-1).astype(int)
    y_pred = np.asarray(y_pred).reshape(-1)

    if y_true.size == 0:
        raise ValueError("The dataset cannot be empty.")

    if y_true.shape != y_pred.shape:
        raise ValueError("Labels and predictions must have matching shapes.")

    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("True labels must contain only 0 or 1.")

    if not np.all(np.isfinite(y_pred)):
        raise ValueError("Predictions must be finite numbers.")

    if not np.all((y_pred >= 0) & (y_pred <= 1)):
        raise ValueError("Predictions must be probabilities between 0 and 1.")

    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")

    predicted_classes = (y_pred >= threshold).astype(int)

    true_positive = np.sum((y_true == 1) & (predicted_classes == 1))
    true_negative = np.sum((y_true == 0) & (predicted_classes == 0))
    false_positive = np.sum((y_true == 0) & (predicted_classes == 1))
    false_negative = np.sum((y_true == 1) & (predicted_classes == 0))

    accuracy = (true_positive + true_negative) / y_true.size

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive > 0 else 0.0
    )

    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative > 0 else 0.0
    )

    f1_score = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0.0
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_score),
        "true_positive": int(true_positive),
        "true_negative": int(true_negative),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
    }
