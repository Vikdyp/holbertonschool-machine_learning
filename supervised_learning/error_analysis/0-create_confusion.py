#!/usr/bin/env python3
"""Create a confusion matrix from one-hot labels and predictions."""

import numpy as np


def create_confusion_matrix(labels, logits):
    """Return a confusion matrix for the supplied labels and logits."""
    classes = labels.shape[1]
    confusion = np.zeros((classes, classes))
    actual = np.argmax(labels, axis=1)
    predicted = np.argmax(logits, axis=1)

    for label, prediction in zip(actual, predicted):
        confusion[label, prediction] += 1

    return confusion
