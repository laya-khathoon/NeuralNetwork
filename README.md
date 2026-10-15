# Neural Network from Scratch

A simple feedforward neural network implemented from scratch using **Python and NumPy**, without relying on deep learning frameworks such as TensorFlow or PyTorch.

This project demonstrates the fundamental concepts behind neural networks, including forward propagation, backpropagation, activation functions, loss calculation, and gradient descent.

## Features

- **Activation Functions:** Sigmoid and ReLU, including their derivatives.
- **Dense Layer:** Fully connected layer with weight and bias initialization.
- **Forward Propagation:** Computes predictions from input data.
- **Backpropagation:** Calculates gradients to train the network.
- **Binary Cross-Entropy:** Measures prediction error for binary classification.
- **Gradient Descent:** Updates model parameters to reduce loss.
- **Training Test:** Demonstrates learning using the OR logic dataset.

## Technologies Used

- Python
- NumPy
- Matplotlib
- Scikit-learn
- Streamlit

## Project Structure

```text
NeuralNetwork/
├── activations.py
├── layers.py
├── losses.py
├── model.py
├── test_activations.py
├── test_layers.py
├── test_losses.py
├── test_training.py
├── .gitignore
└── README.md
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/laya-khathoon/NeuralNetwork.git
cd NeuralNetwork
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install numpy matplotlib scikit-learn streamlit
```

## Run the Project

Train the neural network on the OR logic dataset:

```bash
python test_training.py
```

The program displays the initial loss, training progress, final loss, and predictions.

## Example Results

The network was tested on the OR logic dataset:

| Input | Expected Output |
|---|---:|
| `[0, 0]` | 0 |
| `[0, 1]` | 1 |
| `[1, 0]` | 1 |
| `[1, 1]` | 1 |

In the recorded training run, the loss decreased from approximately **0.992725** to **0.004776**, and all four inputs were classified correctly using a threshold of 0.5.

## Run the Tests

Execute the individual test scripts:

```bash
python test_activations.py
python test_layers.py
python test_losses.py
python test_training.py
```

## Learning Objectives

This project helps demonstrate:

- How neural network layers process input data.
- How activation functions introduce non-linearity.
- How prediction errors are calculated.
- How backpropagation computes gradients.
- How gradient descent updates weights and biases.
- How a neural network learns from training data.

## Future Improvements

- Add accuracy, precision, recall, and F1-score evaluation.
- Visualize the training loss using Matplotlib.
- Support additional activation functions and loss functions.
- Add a Streamlit web interface for interactive predictions.
- Extend the model to multiclass classification.
- Add automated tests and improve documentation.

## Author

**Laya Khathoon V**

## License

This project is available for educational and learning purposes. A formal open-source license can be added when required.