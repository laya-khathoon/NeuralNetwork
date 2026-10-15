
import numpy as np
from layers import DenseLayer

layer = DenseLayer(input_size=2, output_size=3, seed=42)

inputs = np.array([[1.0, 2.0]])
output = layer.forward(inputs)

print("Input shape:", inputs.shape)
print("Weight shape:", layer.weights.shape)
print("Bias shape:", layer.biases.shape)
print("Output shape:", output.shape)
print("Output:", output)

assert layer.weights.shape == (2, 3)
assert layer.biases.shape == (1, 3)
assert output.shape == (1, 3)
assert np.all(np.isfinite(output))

print("\nAll layer tests passed.")
