#!/usr/bin/env python3
"""Calculate class F1 scores from a confusion matrix."""

sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """Return the F1 score of every class in a confusion matrix."""
    class_sensitivity = sensitivity(confusion)
    class_precision = precision(confusion)
    return (
        2 * class_precision * class_sensitivity
        / (class_precision + class_sensitivity)
    )
