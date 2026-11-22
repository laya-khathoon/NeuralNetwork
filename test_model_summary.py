import unittest

from model import NeuralNetwork


class ModelSummaryTests(unittest.TestCase):
    def test_summary_reports_layer_shapes_and_counts(self):
        network = NeuralNetwork(input_size=4, hidden_size=8, seed=1)
        text = network.summary()
        self.assertIn("Hidden layer: 4 -> 8 neurons, 40 parameters", text)
        self.assertIn("Output layer: 8 -> 1 neurons, 9 parameters", text)
        self.assertIn("Total trainable parameters: 49", text)

    def test_summary_counts_match_actual_arrays(self):
        network = NeuralNetwork(input_size=3, hidden_size=5, seed=2)
        expected = (
            network.hidden_layer.weights.size
            + network.hidden_layer.biases.size
            + network.output_layer.weights.size
            + network.output_layer.biases.size
        )
        self.assertIn(f"Total trainable parameters: {expected}",
                      network.summary())

    def test_summary_does_not_change_weights(self):
        network = NeuralNetwork(input_size=2, hidden_size=3, seed=3)
        before = network.hidden_layer.weights.copy()
        network.summary()
        self.assertTrue((network.hidden_layer.weights == before).all())


if __name__ == "__main__":
    unittest.main()
