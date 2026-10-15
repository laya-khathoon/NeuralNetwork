import numpy as np

from model import NeuralNetwork
from losses import binary_cross_entropy


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


# Create the neural network
network = NeuralNetwork(
    input_size=2,
    hidden_size=8,
    output_size=1,
    seed=42
)


# Calculate initial loss
initial_predictions = network.forward(X)
initial_loss = binary_cross_entropy(y, initial_predictions)

print("Initial loss:", round(initial_loss, 6))


# Train the network
epochs = 3000
learning_rate = 0.1

for epoch in range(epochs):
    loss = network.train_step(
        X,
        y,
        learning_rate=learning_rate
    )

    if (epoch + 1) % 500 == 0:
        print(
            f"Epoch {epoch + 1}/{epochs} - "
            f"Loss: {loss:.6f}"
        )


# Evaluate after training
final_predictions = network.forward(X)
final_loss = binary_cross_entropy(y, final_predictions)

print("\nFinal loss:", round(final_loss, 6))
print("\nPredictions:")

for inputs, prediction in zip(X, final_predictions):
    predicted_class = int(prediction[0] >= 0.5)

    print(
        f"Input: {inputs.astype(int)} | "
        f"Probability: {prediction[0]:.4f} | "
        f"Predicted class: {predicted_class}"
    )


# Verify learning
assert final_loss < initial_loss, (
    "Training failed: final loss did not decrease."
)

print("\nTraining test passed!")