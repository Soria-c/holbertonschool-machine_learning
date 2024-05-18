#!/usr/bin/env python3
"""Normalize"""
import numpy as np


def normalize(X, m, s):
    """
    Normalizes (standardizes) a matrix
    ```
    Parameters:
    -----------
    X: numpy.ndarray(d, nx)
        d is the number of data points
        nx is the number of features
    m: numpy.ndarray(nx,)
        contains the mean of all features of X
    s: numpy.ndarray(nx,)
        contains the standard deviation of all features of X
    Return:
    -------
    normalized X matrix
    """
    return (X - m) / s
