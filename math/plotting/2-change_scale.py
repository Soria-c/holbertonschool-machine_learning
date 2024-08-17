#!/usr/bin/env python3
"""Matplotlib tasks"""
import numpy as np
import matplotlib.pyplot as plt


def change_scale():
    """Function to plot a line graph"""
    x = np.arange(0, 28651, 5730)
    r = np.log(0.5)
    t = 5730
    y = np.exp((r / t) * x)
    plt.figure(figsize=(6.4, 4.8))
    
    # your code here
    
    plt.title("Exponential Decay of C-14")
    plt.ylabel("Fraction Remaining")
    plt.xlabel("Time (years)")
    plt.yscale("log")
    plt.gca().set_xlim([0, 28650])
    plt.plot(x, y)
    plt.show()
