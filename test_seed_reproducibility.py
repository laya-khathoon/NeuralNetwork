"""Offline regression test: a fixed seed gives reproducible training.

Anyone following the README should see the same numbers the README reports,
so training with the same seed must be deterministic.
"""
import unittest

import numpy as np

from model import NeuralNetwork

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


def _trained_history(seed):
    network = NeuralNetwork(input_size=2, hidden_size=8, output_size=1, seed=seed)
    return network.fit(X, y, epochs=200, learning_rate=0.1, verbose=False)


class SeedReproducibilityTests(unittest.TestCase):
    def test_same_seed_gives_identical_loss_history(self):
        first = _trained_history(seed=42)
        second = _trained_history(seed=42)
        np.testing.assert_allclose(first, second, rtol=0, atol=0)

    def test_same_seed_gives_identical_weights(self):
        net_a = NeuralNetwork(input_size=2, hidden_size=8, seed=42)
        net_b = NeuralNetwork(input_size=2, hidden_size=8, seed=42)
        np.testing.assert_array_equal(net_a.hidden_layer.weights, net_b.hidden_layer.weights)
        np.testing.assert_array_equal(net_a.output_layer.weights, net_b.output_layer.weights)

    def test_different_seeds_train_differently(self):
        self.assertNotEqual(_trained_history(seed=42)[-1], _trained_history(seed=7)[-1])


if __name__ == "__main__":
    unittest.main()
