import numpy as np

from layers import DenseLayer


def test_update_moves_weights_against_the_gradient():
    layer = DenseLayer(2, 2, seed=3)
    layer.forward(np.ones((3, 2)))
    layer.backward(np.ones((3, 2)))
    before = layer.weights.copy()
    layer.update_parameters(0.1)
    np.testing.assert_allclose(layer.weights, before - 0.1 * layer.dweights)


def test_zero_learning_rate_changes_nothing():
    layer = DenseLayer(2, 2, seed=3)
    layer.forward(np.ones((3, 2)))
    layer.backward(np.ones((3, 2)))
    before = layer.weights.copy()
    layer.update_parameters(0.0)
    np.testing.assert_array_equal(layer.weights, before)
