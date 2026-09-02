#!/usr/bin/env python3
"""Tests for the Convolutions and Pooling project."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np


PROJECT_DIR = (
    Path(__file__).resolve().parents[1]
    / "math"
    / "convolutions_and_pooling"
)


def load_module(filename):
    """Load a task module from the project directory."""
    path = PROJECT_DIR / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestConvolutionsAndPooling(unittest.TestCase):
    """Validate convolution and pooling implementations."""

    def test_valid_convolution_batches_images(self):
        """Convolve every image while leaving batch work vectorized."""
        module = load_module("0-convolve_grayscale_valid.py")
        images = np.arange(32).reshape(2, 4, 4)
        kernel = np.array([[1, 0], [0, -1]])

        result = module.convolve_grayscale_valid(images, kernel)

        expected = np.full((2, 3, 3), -5)
        np.testing.assert_array_equal(result, expected)

    def test_same_convolution_preserves_image_dimensions(self):
        """Use zero padding and preserve height and width for even kernels."""
        module = load_module("1-convolve_grayscale_same.py")
        images = np.arange(1, 13).reshape(1, 3, 4)
        kernel = np.ones((2, 2))

        result = module.convolve_grayscale_same(images, kernel)

        expected = np.array(
            [[[1, 3, 5, 7], [6, 14, 18, 22], [14, 30, 34, 38]]]
        )
        np.testing.assert_array_equal(result, expected)

    def test_strided_convolution_supports_same_and_valid_padding(self):
        """Respect strides while applying named padding modes."""
        module = load_module("3-convolve_grayscale.py")
        images = np.ones((1, 3, 4))
        kernel = np.ones((2, 3))

        same = module.convolve_grayscale(
            images, kernel, padding="same", stride=(2, 2)
        )
        valid = module.convolve_grayscale(
            images, kernel, padding="valid", stride=(2, 1)
        )

        expected_same = np.array(
            [[[0, 0, 0, 0], [0, 4, 6, 2], [0, 2, 3, 1]]]
        )
        np.testing.assert_array_equal(same, expected_same)
        np.testing.assert_array_equal(valid, np.full((1, 1, 2), 6))

    def test_custom_padding_expands_convolution_output(self):
        """Apply independent symmetric height and width padding."""
        module = load_module("2-convolve_grayscale_padding.py")
        images = np.ones((1, 2, 2))
        kernel = np.ones((2, 2))

        result = module.convolve_grayscale_padding(
            images, kernel, (1, 2)
        )

        expected = np.array(
            [[[0, 1, 2, 1, 0],
              [0, 2, 4, 2, 0],
              [0, 1, 2, 1, 0]]]
        )
        np.testing.assert_array_equal(result, expected)

    def test_channel_convolution_reduces_all_input_channels(self):
        """Combine channel contributions into one output feature map."""
        module = load_module("4-convolve_channels.py")
        images = np.arange(1, 9).reshape(1, 2, 2, 2)
        kernel = np.ones((2, 2, 2))

        result = module.convolve_channels(images, kernel)

        expected = np.array(
            [[[3, 10, 7], [14, 36, 22], [11, 26, 15]]]
        )
        np.testing.assert_array_equal(result, expected)

    def test_multiple_kernels_create_multiple_output_channels(self):
        """Apply every kernel and retain a separate result channel."""
        module = load_module("5-convolve.py")
        images = np.arange(1, 5).reshape(1, 2, 2, 1)
        kernels = np.array(
            [[[[1, 1]], [[1, 0]]], [[[1, 0]], [[1, -1]]]]
        )

        result = module.convolve(images, kernels, padding="valid")

        expected = np.array([[[[10, -3]]]])
        np.testing.assert_array_equal(result, expected)

    def test_pool_supports_maximum_and_average_modes(self):
        """Pool spatial windows independently for every image channel."""
        module = load_module("6-pool.py")
        images = np.arange(1, 25).reshape(1, 3, 4, 2)

        maximum = module.pool(images, (2, 2), (1, 2), mode="max")
        average = module.pool(images, (2, 2), (1, 2), mode="avg")

        expected_max = np.array(
            [[[[11, 12], [15, 16]], [[19, 20], [23, 24]]]]
        )
        expected_avg = np.array(
            [[[[6, 7], [10, 11]], [[14, 15], [18, 19]]]]
        )
        np.testing.assert_array_equal(maximum, expected_max)
        np.testing.assert_array_equal(average, expected_avg)


if __name__ == "__main__":
    unittest.main()
