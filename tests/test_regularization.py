#!/usr/bin/env python3
"""Tests for the Regularization project."""

import importlib.util
from pathlib import Path
import sys
import types
import unittest

import numpy as np


PROJECT_DIR = (
    Path(__file__).resolve().parents[1]
    / "supervised_learning"
    / "regularization"
)


def load_module(filename):
    """Load a task module from the Regularization directory."""
    path = PROJECT_DIR / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestRegularization(unittest.TestCase):
    """Validate Regularization task behavior."""

    def test_l2_cost_regularizes_weights_but_not_biases(self):
        """Catch omitted weight layers or accidental bias regularization."""
        filename = "0-l2_reg_cost.py"
        self.assertTrue(
            (PROJECT_DIR / filename).is_file(), f"{filename} missing"
        )
        module = load_module(filename)
        weights = {
            "W1": np.array([[1.0, -2.0], [0.0, 3.0]]),
            "b1": np.full((2, 1), 100.0),
            "W2": np.array([[-1.0, 2.0]]),
            "b2": np.full((1, 1), 100.0),
        }

        result = module.l2_reg_cost(2.0, 0.4, weights, 2, 4)

        self.assertAlmostEqual(result, 2.95)

    def test_l2_gradient_descent_updates_weights_and_biases_in_place(self):
        """Catch wrong L2 scaling or regularization of biases."""
        filename = "1-l2_reg_gradient_descent.py"
        self.assertTrue(
            (PROJECT_DIR / filename).is_file(), f"{filename} missing"
        )
        module = load_module(filename)
        weights = {
            "W1": np.array([[0.5, -0.5], [1.0, 0.0]]),
            "b1": np.zeros((2, 1)),
        }
        cache = {
            "A0": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "A1": np.array([[0.8, 0.3], [0.2, 0.7]]),
        }
        labels = np.array([[1.0, 0.0], [0.0, 1.0]])

        result = module.l2_reg_gradient_descent(
            labels, weights, cache, 0.1, 0.2, 1
        )

        self.assertIsNone(result)
        np.testing.assert_allclose(
            weights["W1"],
            [[0.475, -0.525], [1.01, 0.03]],
        )
        np.testing.assert_allclose(weights["b1"], [[-0.005], [0.005]])

    def test_tensorflow_l2_cost_returns_one_total_per_regularized_layer(self):
        """Catch reduction of all layer penalties into one scalar."""
        filename = "2-l2_reg_cost.py"
        self.assertTrue(
            (PROJECT_DIR / filename).is_file(), f"{filename} missing"
        )
        source = (PROJECT_DIR / filename).read_text(encoding="utf-8")
        self.assertIn("import tensorflow as tf", source)
        old_tensorflow = sys.modules.get("tensorflow")
        sys.modules["tensorflow"] = types.SimpleNamespace()
        try:
            module = load_module(filename)
        finally:
            if old_tensorflow is None:
                sys.modules.pop("tensorflow", None)
            else:
                sys.modules["tensorflow"] = old_tensorflow

        class ModelWithLosses:
            """Expose the Keras model.losses contract used by the task."""

            losses = [1.0, 2.5, 4.0]

        result = module.l2_reg_cost(np.array(0.5), ModelWithLosses())

        np.testing.assert_allclose(result, [1.5, 3.0, 4.5])

    def test_l2_layer_configures_dense_regularization(self):
        """Catch missing L2 regularizer or incorrect variance scaling mode."""
        filename = "3-l2_reg_create_layer.py"
        self.assertTrue(
            (PROJECT_DIR / filename).is_file(), f"{filename} missing"
        )

        class VarianceScaling:
            """Record the requested initializer mode."""

            def __init__(self, scale, mode):
                self.scale = scale
                self.mode = mode

        class L2:
            """Record the requested L2 coefficient."""

            def __init__(self, coefficient):
                self.coefficient = coefficient

        class Dense:
            """Expose Dense configuration as its observable output."""

            def __init__(self, units, activation, kernel_initializer,
                         kernel_regularizer):
                self.configuration = {
                    "units": units,
                    "activation": activation,
                    "initializer": kernel_initializer,
                    "regularizer": kernel_regularizer,
                }

            def __call__(self, previous):
                """Return the configured layer and its input."""
                return previous, self.configuration

        fake_tensorflow = types.SimpleNamespace(
            keras=types.SimpleNamespace(
                initializers=types.SimpleNamespace(
                    VarianceScaling=VarianceScaling,
                ),
                regularizers=types.SimpleNamespace(L2=L2),
                layers=types.SimpleNamespace(Dense=Dense),
            )
        )
        old_tensorflow = sys.modules.get("tensorflow")
        sys.modules["tensorflow"] = fake_tensorflow
        try:
            module = load_module(filename)
            previous, configuration = module.l2_reg_create_layer(
                "previous", 7, "relu", 0.3
            )
        finally:
            if old_tensorflow is None:
                sys.modules.pop("tensorflow", None)
            else:
                sys.modules["tensorflow"] = old_tensorflow

        self.assertEqual(previous, "previous")
        self.assertEqual(configuration["units"], 7)
        self.assertEqual(configuration["activation"], "relu")
        self.assertEqual(configuration["initializer"].scale, 2.0)
        self.assertEqual(configuration["initializer"].mode, "fan_avg")
        self.assertEqual(configuration["regularizer"].coefficient, 0.3)

    def test_dropout_forward_prop_masks_hidden_layers_only(self):
        """Catch missing inverted-dropout scaling or output-layer dropout."""
        filename = "4-dropout_forward_prop.py"
        self.assertTrue(
            (PROJECT_DIR / filename).is_file(), f"{filename} missing"
        )
        module = load_module(filename)
        inputs = np.array([[1.0, 2.0], [0.0, 1.0]])
        weights = {
            "W1": np.eye(2),
            "b1": np.zeros((2, 1)),
            "W2": np.eye(2),
            "b2": np.zeros((2, 1)),
        }
        np.random.seed(1)

        cache = module.dropout_forward_prop(inputs, weights, 2, 0.5)

        np.testing.assert_array_equal(cache["D1"], [[0, 1], [0, 0]])
        np.testing.assert_allclose(
            cache["A1"],
            [[0.0, 1.9280551601516338], [0.0, 0.0]],
        )
        np.testing.assert_allclose(
            cache["A2"],
            [[0.5, 0.8730339992227998], [0.5, 0.1269660007772002]],
        )
        self.assertNotIn("D2", cache)

    def test_dropout_gradient_descent_applies_hidden_mask_and_scaling(self):
        """Catch omitted mask application or keep-probability scaling."""
        filename = "5-dropout_gradient_descent.py"
        self.assertTrue(
            (PROJECT_DIR / filename).is_file(), f"{filename} missing"
        )
        module = load_module(filename)
        labels = np.array([[1.0, 0.0], [0.0, 1.0]])
        weights = {
            "W1": np.array([[0.1, 0.2], [0.3, 0.4]]),
            "b1": np.zeros((2, 1)),
            "W2": np.array([[0.5, -0.2], [-0.3, 0.4]]),
            "b2": np.zeros((2, 1)),
        }
        cache = {
            "A0": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "A1": np.array([[0.2, 0.0], [0.4, -0.6]]),
            "D1": np.array([[1, 0], [1, 1]]),
            "A2": np.array([[0.7, 0.1], [0.3, 0.9]]),
        }

        result = module.dropout_gradient_descent(
            labels, weights, cache, 0.1, 0.5, 2
        )

        self.assertIsNone(result)
        np.testing.assert_allclose(
            weights["W1"], [[0.12304, 0.26912], [0.29256, 0.37]]
        )
        np.testing.assert_allclose(weights["b1"], [[0.02304], [-0.01128]])
        np.testing.assert_allclose(
            weights["W2"], [[0.503, -0.191], [-0.303, 0.391]]
        )
        np.testing.assert_allclose(weights["b2"], [[0.01], [-0.01]])

    def test_dropout_layer_uses_inverted_rate_and_training_flag(self):
        """Catch use of keep probability as the Keras dropout rate."""
        filename = "6-dropout_create_layer.py"
        self.assertTrue(
            (PROJECT_DIR / filename).is_file(), f"{filename} missing"
        )

        class VarianceScaling:
            """Record the requested initializer mode."""

            def __init__(self, mode):
                self.mode = mode

        class Dense:
            """Return an observable dense-layer description."""

            def __init__(self, units, activation, kernel_initializer):
                self.units = units
                self.activation = activation
                self.initializer = kernel_initializer

            def __call__(self, previous):
                """Return the configured dense output."""
                return {
                    "previous": previous,
                    "units": self.units,
                    "activation": self.activation,
                    "initializer": self.initializer,
                }

        class Dropout:
            """Return an observable dropout-layer description."""

            def __init__(self, rate):
                self.rate = rate

            def __call__(self, value, training):
                """Return dropout configuration and its input."""
                return {
                    "value": value,
                    "rate": self.rate,
                    "training": training,
                }

        fake_tensorflow = types.SimpleNamespace(
            keras=types.SimpleNamespace(
                initializers=types.SimpleNamespace(
                    VarianceScaling=VarianceScaling,
                ),
                layers=types.SimpleNamespace(Dense=Dense, Dropout=Dropout),
            )
        )
        old_tensorflow = sys.modules.get("tensorflow")
        sys.modules["tensorflow"] = fake_tensorflow
        try:
            module = load_module(filename)
            result = module.dropout_create_layer(
                "previous", 5, "tanh", 0.8, training=False
            )
        finally:
            if old_tensorflow is None:
                sys.modules.pop("tensorflow", None)
            else:
                sys.modules["tensorflow"] = old_tensorflow

        self.assertAlmostEqual(result["rate"], 0.2)
        self.assertFalse(result["training"])
        self.assertEqual(result["value"]["previous"], "previous")
        self.assertEqual(result["value"]["units"], 5)
        self.assertEqual(result["value"]["activation"], "tanh")
        self.assertEqual(result["value"]["initializer"].mode, "fan_avg")

    def test_early_stopping_resets_or_advances_patience(self):
        """Catch inclusive-threshold or off-by-one patience errors."""
        filename = "7-early_stopping.py"
        self.assertTrue(
            (PROJECT_DIR / filename).is_file(), f"{filename} missing"
        )
        module = load_module(filename)

        self.assertEqual(
            module.early_stopping(1.0, 1.9, 0.5, 15, 5),
            (False, 0),
        )
        self.assertEqual(
            module.early_stopping(1.0, 1.5, 0.5, 15, 8),
            (False, 9),
        )
        self.assertEqual(
            module.early_stopping(1.0, 1.5, 0.5, 15, 14),
            (True, 15),
        )


if __name__ == "__main__":
    unittest.main()
