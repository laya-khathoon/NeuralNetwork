from model import NeuralNetwork


def test_summary_total_matches_layer_sizes():
    net = NeuralNetwork(input_size=2, hidden_size=4, output_size=1, seed=1)
    # hidden: 2*4 + 4 = 12, output: 4*1 + 1 = 5
    assert "Total trainable parameters: 17" in net.summary()


def test_summary_names_both_layers():
    text = NeuralNetwork(3, 5, seed=1).summary()
    assert "Hidden layer: 3 -> 5 neurons" in text
    assert "Output layer: 5 -> 1 neurons" in text
