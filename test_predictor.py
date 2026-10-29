import numpy as np
import pytest

from predictor import predict


class DummyNetwork:
    def forward(self, X):
        return np.array([[0.2], [0.8]])[:len(X)]


def test_predict_multiple_samples():
    probabilities, classes = predict(
        DummyNetwork(),
        np.array([[1, 2], [3, 4]])
    )

    np.testing.assert_allclose(probabilities, [[0.2], [0.8]])
    np.testing.assert_array_equal(classes, [[0], [1]])


def test_predict_single_sample():
    class SingleSampleNetwork:
        def forward(self, X):
            return np.array([[0.8]])

    probabilities, classes = predict(
        SingleSampleNetwork(),
        [1, 2]
    )

    assert probabilities.shape == (1, 1)
    np.testing.assert_array_equal(classes, [[1]])


def test_custom_threshold():
    _, classes = predict(
        DummyNetwork(),
        [[1, 2], [3, 4]],
        threshold=0.9
    )

    np.testing.assert_array_equal(classes, [[0], [0]])


def test_invalid_threshold():
    with pytest.raises(ValueError):
        predict(DummyNetwork(), [[1, 2]], threshold=1.5)


def test_empty_input():
    with pytest.raises(ValueError):
        predict(DummyNetwork(), np.empty((0, 2)))
