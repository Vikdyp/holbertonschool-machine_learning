#!/usr/bin/env python3
"""Tests for the Error Analysis project."""

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


PROJECT_DIR = (
    Path(__file__).resolve().parents[1]
    / "supervised_learning"
    / "error_analysis"
)
sys.path.insert(0, str(PROJECT_DIR))


def load_module(filename):
    """Load a task module from the Error Analysis directory."""
    path = PROJECT_DIR / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestErrorAnalysis(unittest.TestCase):
    """Validate Error Analysis task behavior."""

    def test_confusion_matrix_counts_actual_rows_and_predicted_columns(self):
        """Catch swapped axes or incorrect one-hot aggregation."""
        self.assertTrue(
            (PROJECT_DIR / "0-create_confusion.py").is_file(),
            "0-create_confusion.py is missing",
        )
        module = load_module("0-create_confusion.py")
        labels = np.eye(3)[[0, 1, 2, 1]]
        logits = np.eye(3)[[0, 2, 2, 0]]
        expected = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ])

        result = module.create_confusion_matrix(labels, logits)

        np.testing.assert_array_equal(result, expected)

    def test_sensitivity_uses_each_actual_class_total(self):
        """Catch use of predicted totals instead of actual totals."""
        self.assertTrue(
            (PROJECT_DIR / "1-sensitivity.py").is_file(),
            "1-sensitivity.py is missing",
        )
        module = load_module("1-sensitivity.py")
        confusion = np.array([
            [8.0, 2.0, 0.0],
            [1.0, 6.0, 3.0],
            [0.0, 2.0, 8.0],
        ])

        result = module.sensitivity(confusion)

        np.testing.assert_allclose(result, [0.8, 0.6, 0.8])

    def test_precision_uses_each_predicted_class_total(self):
        """Catch use of actual totals instead of predicted totals."""
        self.assertTrue(
            (PROJECT_DIR / "2-precision.py").is_file(),
            "2-precision.py is missing",
        )
        module = load_module("2-precision.py")
        confusion = np.array([
            [8.0, 2.0, 0.0],
            [1.0, 6.0, 3.0],
            [0.0, 2.0, 8.0],
        ])

        result = module.precision(confusion)

        np.testing.assert_allclose(result, [8 / 9, 0.6, 8 / 11])

    def test_specificity_uses_true_negatives_and_false_positives(self):
        """Catch incorrect negative-class totals."""
        self.assertTrue(
            (PROJECT_DIR / "3-specificity.py").is_file(),
            "3-specificity.py is missing",
        )
        module = load_module("3-specificity.py")
        confusion = np.array([
            [8.0, 2.0, 0.0],
            [1.0, 6.0, 3.0],
            [0.0, 2.0, 8.0],
        ])

        result = module.specificity(confusion)

        np.testing.assert_allclose(result, [0.95, 0.8, 0.85])

    def test_f1_score_balances_precision_and_sensitivity(self):
        """Catch omission of either precision or sensitivity."""
        self.assertTrue(
            (PROJECT_DIR / "4-f1_score.py").is_file(),
            "4-f1_score.py is missing",
        )
        module = load_module("4-f1_score.py")
        confusion = np.array([
            [8.0, 2.0, 0.0],
            [1.0, 6.0, 3.0],
            [0.0, 2.0, 8.0],
        ])

        result = module.f1_score(confusion)

        np.testing.assert_allclose(result, [16 / 19, 0.6, 16 / 21])


if __name__ == "__main__":
    unittest.main()
