#!/usr/bin/env python3
"""Check CNN values, overlapping windows, and numerical gradients."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np


PROJECT = Path(__file__).resolve().parents[1] / "supervised_learning/cnn"


def load_function(filename, name):
    """Load a numbered exercise and report missing implementations clearly."""
    path = PROJECT / filename
    assert path.is_file(), "Missing implementation: {}".format(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, name)


def numerical_gradient(function, values):
    """Estimate derivatives by centered finite differences."""
    gradient = np.zeros_like(values)
    epsilon = 1e-6
    for index in np.ndindex(values.shape):
        original = values[index]
        values[index] = original + epsilon
        plus = function()
        values[index] = original - epsilon
        minus = function()
        values[index] = original
        gradient[index] = (plus - minus) / (2 * epsilon)
    return gradient


class CNNTests(unittest.TestCase):
    """Catch channel, padding, stride, and gradient accumulation errors."""

    def test_0_convolution_channels_bias_activation_and_stride(self):
        """A missing channel sum or misplaced bias changes literal outputs."""
        forward = load_function("0-conv_forward.py", "conv_forward")
        values = np.arange(1, 19).reshape(1, 3, 3, 2) / 100
        weights = np.ones((2, 2, 2, 2))
        weights[..., 1] = -0.5
        bias = np.array([.01, -.02]).reshape(1, 1, 1, 2)
        output = forward(values, weights, bias, np.tanh, padding="valid")
        expected = np.array([[[[45, -24], [61, -32]],
                              [[93, -48], [109, -56]]]])
        np.testing.assert_allclose(output, np.tanh(expected / 100))
        output = forward(np.ones((1, 3, 3, 1)),
                         np.ones((3, 3, 1, 1)), np.zeros((1, 1, 1, 1)),
                         lambda x: x, padding="same", stride=(2, 2))
        np.testing.assert_array_equal(
            output[0, :, :, 0], [[1, 3, 1], [3, 9, 3], [1, 3, 1]])

    def test_1_pooling_modes_and_rectangular_stride(self):
        """Max and average pooling must preserve channels and ignore tails."""
        forward = load_function("1-pool_forward.py", "pool_forward")
        values = np.arange(1, 21).reshape(1, 4, 5, 1)
        maximum = forward(values, (2, 3), stride=(2, 2))
        average = forward(values, (2, 3), stride=(2, 2), mode="avg")
        np.testing.assert_array_equal(maximum[0, :, :, 0], [[8, 10], [18, 20]])
        np.testing.assert_array_equal(average[0, :, :, 0],
                                      [[4.5, 6.5], [14.5, 16.5]])

    def test_2_convolution_gradients_match_finite_differences(self):
        """Wrong padding crops or overwritten overlaps break derivatives."""
        forward = load_function("0-conv_forward.py", "conv_forward")
        backward = load_function("2-conv_backward.py", "conv_backward")
        rng = np.random.default_rng(11)
        for padding, kernel, stride in [
                ("valid", (2, 2), (1, 2)),
                ("same", (3, 2), (2, 1)),
                ("same", (1, 1), (1, 1))]:
            with self.subTest(padding=padding, kernel=kernel, stride=stride):
                values = rng.normal(size=(2, 3, 4, 2))
                weights = rng.normal(size=(*kernel, 2, 2))
                bias = rng.normal(size=(1, 1, 1, 2))
                output = forward(values, weights, bias, lambda x: x,
                                 padding=padding, stride=stride)
                upstream = rng.normal(size=output.shape)

                def loss():
                    return np.sum(forward(values, weights, bias, lambda x: x,
                                          padding, stride) * upstream)

                actual = backward(upstream, values, weights, bias,
                                  padding, stride)
                for derivative, parameter in zip(
                        actual, (values, weights, bias)):
                    np.testing.assert_allclose(
                        derivative, numerical_gradient(loss, parameter),
                        atol=1e-7, rtol=1e-6)

    def test_3_pooling_gradient_overlaps_and_tied_maxima(self):
        """Contributions from overlapping windows must accumulate."""
        backward = load_function("3-pool_backward.py", "pool_backward")
        values = np.array([[1, 2, 3], [4, 9, 6], [7, 8, 5]],
                          dtype=float).reshape(1, 3, 3, 1)
        upstream = np.ones((1, 2, 2, 1))
        actual = backward(upstream, values, (2, 2))
        np.testing.assert_array_equal(actual[0, :, :, 0],
                                      [[0, 0, 0], [0, 4, 0], [0, 0, 0]])
        actual = backward(upstream, values, (2, 2), mode="avg")
        np.testing.assert_array_equal(
            actual[0, :, :, 0], [[.25, .5, .25], [.5, 1, .5], [.25, .5, .25]])
        tied = backward(np.ones((1, 1, 1, 1)), np.ones((1, 2, 2, 1)), (2, 2))
        np.testing.assert_array_equal(tied, np.ones((1, 2, 2, 1)))

    @unittest.skipUnless(importlib.util.find_spec("tensorflow"),
                         "TensorFlow is not installed")
    def test_4_lenet_predicts_ten_classes_and_trains(self):
        """The specified classifier must be reproducible and trainable."""
        from tensorflow import keras as K
        build = load_function("5-lenet5.py", "lenet5")
        model = build(K.Input(shape=(28, 28, 1)))
        self.assertEqual(model.output_shape, (None, 10))
        self.assertEqual(model.count_params(), 61706)
        images = np.zeros((2, 28, 28, 1), dtype=np.float32)
        predictions = model(images).numpy()
        np.testing.assert_allclose(predictions.sum(axis=1), [1, 1])
        second = build(K.Input(shape=(28, 28, 1)))
        for left, right in zip(model.get_weights(), second.get_weights()):
            np.testing.assert_array_equal(left, right)
        result = model.train_on_batch(images, np.eye(10)[[0, 1]])
        self.assertTrue(np.all(np.isfinite(result)))


if __name__ == "__main__":
    unittest.main()
