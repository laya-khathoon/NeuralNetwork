
import numpy as np

from model import NeuralNetwork
from visualization import plot_training_loss


# OR logic dataset
X = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
])

y = np.array([
    [0.0],
    [1.0],
    [1.0],
    [1.0]
])


# Create and train the network
network = NeuralNetwork(
    input_size=2,
    hidden_size=8,
    output_size=1,
    seed=42
)

history = network.fit(
    X,
    y,
    epochs=3000,
    learning_rate=0.1,
    verbose=False
)

# Plot the actual recorded training loss
plot_training_loss(
    history,
    save_path="training_loss.png"
)

print("Training completed successfully.")
print(f"Recorded epochs: {len(history)}")
print(f"Initial loss: {history[0]:.6f}")
print(f"Final loss: {history[-1]:.6f}")
print("Graph saved as training_loss.png")
