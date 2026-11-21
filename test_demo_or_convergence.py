"""Offline regression test: the OR demo network actually learns.

Mirrors demo_training_plot.py: same dataset, architecture and seed, so this
test guards the exact setup the demo script shows.
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


class DemoOrConvergenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.network = NeuralNetwork(input_size=2, hidden_size=8, output_size=1, seed=42)
        cls.history = cls.network.fit(X, y, epochs=3000, learning_rate=0.1, verbose=False)

    def test_loss_decreases_over_training(self):
        self.assertEqual(len(self.history), 3000)
        self.assertLess(self.history[-1], self.history[0])

    def test_final_loss_is_small(self):
        self.assertLess(self.history[-1], 0.05)

    def test_or_predictions_are_correct(self):
        predictions = (self.network.forward(X) >= 0.5).astype(int)
        np.testing.assert_array_equal(predictions, y.astype(int))


if __name__ == "__main__":
    unittest.main()
