#!/usr/bin/env python3
"""This module defines a function"""
import numpy as np


def one_hot_encode(Y, classes):
    """
    Converts a numeric label vector into a one-hot matrix.
    ```
    Parameters:
    -----------
    Y: numpy.ndarray
        contains numeric class labels
    classes: int
        maximum number of classes found in Y
    Returns:
    --------
        a one-hot encoding of Y with shape (classes, m), or None on failure
    """
    try:
        return np.identity(classes)[Y].transpose()
    except Exception:
        return None
