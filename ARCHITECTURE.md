# Architecture

How data flows through the from-scratch network in `model.py`, and where each
piece lives. Everything is plain NumPy; there is no deep-learning framework.

## Pieces

| File | Class / function | Job |
| --- | --- | --- |
| `layers.py` | `DenseLayer` | One fully connected layer: weights, biases, forward, backward, gradient-descent update. |
| `activations.py` | `sigmoid`, `relu`, `sigmoid_derivative`, `relu_derivative` | Non-linearities and their derivatives. |
| `losses.py` | `binary_cross_entropy` | Loss for binary classification. |
| `model.py` | `NeuralNetwork` | Wires two `DenseLayer`s into input -> hidden (ReLU) -> output (sigmoid). |
| `predictor.py` | `predict` | Turns probabilities into class labels with a threshold. |
| `evaluation.py` | `evaluate_classification` | Accuracy, precision, recall and F1. |

## Forward pass

1. `hidden_z = X @ W1 + b1` (`DenseLayer.forward` on the hidden layer)
2. `hidden = relu(hidden_z)`
3. `output_z = hidden @ W2 + b2` (`DenseLayer.forward` on the output layer)
4. `predictions = sigmoid(output_z)` - one probability per sample

## Backward pass

1. `doutput = predictions - y` - derivative of binary cross-entropy with a
   sigmoid output
2. `DenseLayer.backward` on the output layer turns `doutput` into weight and
   bias gradients and passes `dinputs` upstream
3. `dinputs` is multiplied by `relu_derivative(hidden_z)` so dead ReLU units
   get zero gradient
4. `DenseLayer.backward` on the hidden layer computes its gradients

## Update

`update_parameters(learning_rate)` applies plain gradient descent:
`param -= learning_rate * gradient`. One `train_step` runs forward, loss,
backward and update; `fit` repeats it and records the loss history.

## Weight initialization

Hidden and output weights use He initialization: samples from a standard
normal scaled by `sqrt(2 / input_size)`. Biases start at zero. Seeds make
runs reproducible (see `test_seed_reproducibility.py`).
