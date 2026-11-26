import numpy as np

from layers import DenseLayer


def test_same_seed_gives_same_weights():
    first = DenseLayer(3, 3, seed=11)
    second = DenseLayer(3, 3, seed=11)
    np.testing.assert_array_equal(first.weights, second.weights)


def test_different_seeds_give_different_weights():
    first = DenseLayer(3, 3, seed=11)
    second = DenseLayer(3, 3, seed=12)
    assert not np.array_equal(first.weights, second.weights)
