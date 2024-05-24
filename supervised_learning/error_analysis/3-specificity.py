#!/usr/bin/env python3
"""Specificity"""
import numpy as np


def specificity(confusion):
    """
    Calculates the specificity for each class in a confusion matrix
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
    TP = np.diag(confusion)
    TN = confusion.sum() - confusion.sum(axis=0) - confusion.sum(axis=1) + TP
    return TN / (TN + (confusion.sum(axis=0) - TP))
