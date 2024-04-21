#!/usr/bin/env python3
"""Matplotlib tasks"""
import numpy as np
import matplotlib.pyplot as plt


def two():
    """Function to plot two line graphs"""
    x = np.arange(0, 21000, 1000)
    r = np.log(0.5)
    t1 = 5730
    t2 = 1600
    y1 = np.exp((r / t1) * x)
    y2 = np.exp((r / t2) * x)
    plt.figure(figsize=(6.4, 4.8))
    plt.title("Exponential Decay of Radioactive Elements")
    plt.ylabel("Fraction Remaining")
    plt.xlabel("Time (years)")
    plt.plot(x, y1, "r--", x, y2, "g-")
    plt.legend(["C-14", "Ra-226"])
    plt.axis((0, 20000, 0, 1))
    plt.show()
