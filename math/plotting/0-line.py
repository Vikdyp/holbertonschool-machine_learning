#!/usr/bin/env python3
"""module that define the line function"""
import numpy as np
import matplotlib.pyplot as plt


def line():
    """function that print a line with matplotlib"""

    y = np.arange(0, 11) ** 3
    x = np.arange(0, 11)
    plt.figure(figsize=(6.4, 4.8))
    plt.plot(x, y, 'r-')
    plt.xlim(0, 10)
    plt.show()
