
import numpy as np

from model import NeuralNetwork


X = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
])

y = np.array([[0.0], [1.0], [1.0], [1.0]])

network = NeuralNetwork(input_size=2, hidden_size=8, seed=42)

history = network.fit(
    X,
    y,
    epochs=3000,
    learning_rate=0.1,
    verbose=True
)

print("\nRecorded epochs:", len(history))
print("Initial recorded loss:", round(history[0], 6))
print("Final recorded loss:", round(history[-1], 6))

assert len(history) == 3000
assert all(np.isfinite(loss) for loss in history)
assert history[-1] < history[0]

print("\nTraining history test passed!")
