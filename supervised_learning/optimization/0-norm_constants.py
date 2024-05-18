#!/usr/bin/env python3
"""Normalization Constants"""
import numpy as np


def normalization_constants(X):
    """
    Calculates the normalization (standardization) constants of a matrix
    ```
    Parameters:
    -----------
    X: numpy.ndarray(m, nx)
        m is the number of data points
        nx is the number of features
    Return:
    -------
    the mean and standard deviation of each feature, respectively
    """
    return np.mean(X, axis=0), np.std(X, axis=0)
