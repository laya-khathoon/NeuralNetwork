"""The loss must stay finite even for predictions of exactly 0 or 1."""
import math

from losses import binary_cross_entropy

assert math.isfinite(binary_cross_entropy([1, 0], [0.0, 1.0]))
assert math.isfinite(binary_cross_entropy([1, 0], [1.0, 0.0]))
assert binary_cross_entropy([1, 0], [1.0, 0.0]) < 1e-9

print("Loss extreme tests passed.")
