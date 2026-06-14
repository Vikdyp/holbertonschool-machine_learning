#!/usr/bin/env python3
"""Apply stepwise inverse time learning rate decay."""


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """Return alpha decayed in a stepwise inverse time schedule."""
    return alpha / (1 + decay_rate * (global_step // decay_step))
