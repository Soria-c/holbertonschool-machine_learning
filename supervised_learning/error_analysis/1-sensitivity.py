#!/usr/bin/env python3
"""Create Confusion"""
import numpy as np


def sensitivity(confusion):
    """
    Calculates the sensitivity for each class in a confusion matrix
    Parameters:
    -----------
    confusion numpy.ndarray(classes, classes):
        confusion matrix where row indices represent the correct
        labels and column indices represent the predicted labels.
        classes is the number of classes
    Returns:
    --------
        numpy.ndarray(classes,) containing
        the sensitivity of each class
    """
    return np.diag(confusion) / np.sum(confusion, axis=1)
