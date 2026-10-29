
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from model import NeuralNetwork
from evaluation import evaluate_classification
from visualization import plot_training_loss


def main():
    # Load the dataset
    dataset = load_breast_cancer()
    X = dataset.data
    y = dataset.target.reshape(-1, 1)

    print("Dataset:", dataset.DESCR.splitlines()[0])
    print("Number of samples:", X.shape[0])
    print("Number of features:", X.shape[1])

    # Split before scaling to prevent test-data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Fit the scaler only on the training data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train the NumPy neural network
    network = NeuralNetwork(
        input_size=X_train.shape[1],
        hidden_size=16,
        output_size=1,
        seed=42
    )

    history = network.fit(
        X_train,
        y_train,
        epochs=1000,
        learning_rate=0.01,
        verbose=True
    )

    # Evaluate on unseen test data
    probabilities = network.forward(X_test)

    metrics = evaluate_classification(
        y_test,
        probabilities,
        threshold=0.5
    )

    print("\nTest Set Evaluation")
    print("-------------------")

    for name, value in metrics.items():
        print(f"{name}: {value}")

    # Save the actual training-loss graph
    plot_training_loss(
        history,
        save_path="real_dataset_training_loss.png"
    )

    print("\nTraining completed.")
    print("Graph saved as real_dataset_training_loss.png")


if __name__ == "__main__":
    main()
