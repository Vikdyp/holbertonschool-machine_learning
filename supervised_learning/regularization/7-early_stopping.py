#!/usr/bin/env python3
"""Determine when validation performance warrants early stopping."""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """Return whether to stop and the updated patience counter."""
    if opt_cost - cost > threshold:
        count = 0
    else:
        count += 1

    return count >= patience, count
