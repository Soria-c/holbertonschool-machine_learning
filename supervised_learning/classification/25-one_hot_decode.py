#!/usr/bin/env python3
"""This module defines a function"""
import numpy as np


def one_hot_decode(one_hot):
    """
    Converts a one-hot matrix into a vector of labels.
    ```
    Parameters:
    -----------
    one_hot: numpy.ndarray
         is a one-hot encoded numpy.ndarray with shape (classes, m)
    Returns:
    --------
        a numpy.ndarray with shape (m, ) containing
        the numeric labels for each example, or None on failure
    """
    if type(one_hot) is not np.ndarray or len(one_hot.shape) != 2:
        return
    try:
        return np.argmax(one_hot.transpose(), axis=1)
    except Exception:
        return
