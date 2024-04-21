#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
"""Matplotlib tasks"""


def line():
    """Fuction to plot a curve"""
    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))

    x = np.arange(11)
    plt.plot(x, y, "r-")
    plt.show()
