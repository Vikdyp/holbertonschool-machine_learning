#!/usr/bin/env python3
"""Calculate bias-corrected weighted moving averages."""


def moving_average(data, beta):
    """Return the bias-corrected moving average of data."""
    averages = []
    weighted_sum = 0

    for step, value in enumerate(data, 1):
        weighted_sum = beta * weighted_sum + (1 - beta) * value
        averages.append(weighted_sum / (1 - beta ** step))

    return averages
