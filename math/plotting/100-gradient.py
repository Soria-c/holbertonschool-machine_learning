#!/usr/bin/env python3
"""Matplotlib tasks"""
import numpy as np
import matplotlib.pyplot as plt


def gradient():
    """Function to plot a scatter plot with gradient"""
    np.random.seed(5)
    x = np.random.randn(2000) * 10
    y = np.random.randn(2000) * 10
    z = np.random.rand(2000) + 40 - np.sqrt(np.square(x) + np.square(y))
    plt.figure(figsize=(6.4, 4.8))
    
    # your code here
    
    plt.title("Mountain Elevation")
    plt.ylabel("y coordinate (m)")
    plt.xlabel("x coordinate (m)")
    plt.scatter(x, y, c=z)
    plt.colorbar(label="elevation (m)", orientation="vertical")
    plt.show()
