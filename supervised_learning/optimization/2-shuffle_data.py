#!/usr/bin/env python3
"""Shuffle Data"""
import numpy as np


def shuffle_data(X, Y):
    """
    Shuffles the data points in two matrices the same way
    ```
    Parameters:
    -----------
    X: numpy.ndarray(m, nx)
        m is the number of data points
        nx is the number of features
    Y: numpy.ndarray(m, ny)
        m is the same number of data points as in X
        ny is the number of features
    Return:
    -------
    the shuffled X and Y matrices
    """
    shuffled_indices = np.random.permutation(len(X))
    return X[shuffled_indices], Y[shuffled_indices]
