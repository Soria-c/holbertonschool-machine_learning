#!/usr/bin/env python3
"""F1 Score"""
import numpy as np


sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    Calculates the F1 score of a confusion matrix:
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
    PPV = precision(confusion)
    TPR = sensitivity(confusion)
    return (2 * PPV * TPR)\
        / (PPV + TPR)
