#!/usr/bin/env python3
"""Calculate per-layer Keras costs with L2 regularization."""


def l2_reg_cost(cost, model):
    """Return the base cost plus every regularization loss in the model."""
    return cost + model.losses
