"""Property checks for binary cross-entropy."""
import numpy as np

from losses import binary_cross_entropy

y_true = np.array([1, 0, 1, 0])

# A confident correct answer costs less than an unsure one.
assert binary_cross_entropy(y_true, [0.99, 0.01, 0.99, 0.01]) < binary_cross_entropy(y_true, [0.6, 0.4, 0.6, 0.4])

# A confident wrong answer costs more than a coin flip.
assert binary_cross_entropy(y_true, [0.01, 0.99, 0.01, 0.99]) > binary_cross_entropy(y_true, [0.5] * 4)

# A coin flip costs ln(2) for every label.
assert np.isclose(binary_cross_entropy(y_true, [0.5] * 4), np.log(2))

# The loss is never negative.
assert binary_cross_entropy(y_true, [1.0, 0.0, 1.0, 0.0]) >= 0

print("Loss property tests passed.")
