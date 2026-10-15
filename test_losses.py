
from losses import binary_cross_entropy

# Correct labels
y_true = [1, 0, 1]

# Two correct predictions and one less confident prediction
good_predictions = [0.9, 0.1, 0.8]

# Poor predictions
bad_predictions = [0.1, 0.9, 0.2]

good_loss = binary_cross_entropy(y_true, good_predictions)
bad_loss = binary_cross_entropy(y_true, bad_predictions)

print("Loss for good predictions:", good_loss)
print("Loss for bad predictions:", bad_loss)

assert good_loss < bad_loss

print("\nAll loss function tests passed.")
