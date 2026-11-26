import numpy as np

from layers import DenseLayer


def test_forward_output_shape():
    layer = DenseLayer(3, 5, seed=1)
    assert layer.forward(np.ones((4, 3))).shape == (4, 5)


def test_weights_and_biases_shapes():
    layer = DenseLayer(3, 5, seed=1)
    assert layer.weights.shape == (3, 5)
    assert layer.biases.shape == (1, 5)


def test_biases_start_at_zero():
    layer = DenseLayer(4, 2, seed=1)
    assert not layer.biases.any()
