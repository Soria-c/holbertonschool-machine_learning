#!/usr/bin/env python3
"""Correlation"""

import numpy as np


def correlation(C):
    """
    Calculates a correlation matrix
    """
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    stds_diag = np.sqrt(np.diag(C))
    return C / np.outer(stds_diag, stds_diag)
