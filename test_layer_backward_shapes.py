import numpy as np

from layers import DenseLayer


def test_backward_gradient_shapes_match_parameters():
    layer = DenseLayer(3, 2, seed=7)
    layer.forward(np.ones((5, 3)))
    dinputs = layer.backward(np.ones((5, 2)))
    assert layer.dweights.shape == layer.weights.shape
    assert layer.dbiases.shape == layer.biases.shape
    assert dinputs.shape == (5, 3)


def test_bias_gradient_is_the_mean_over_samples():
    layer = DenseLayer(2, 1, seed=7)
    layer.forward(np.zeros((4, 2)))
    layer.backward(np.array([[1.0], [2.0], [3.0], [4.0]]))
    np.testing.assert_allclose(layer.dbiases, [[2.5]])
