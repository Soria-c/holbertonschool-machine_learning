#!/usr/bin/env python3
"""Principal Components Analysis"""
import numpy as np


def pca(X, ndim):
    """
    Performs PCA on a dataset
    """
    x_m = X - np.mean(X, axis=0)

    U, S, Vt = np.linalg.svd(x_m)

    prin_comp = Vt[:ndim].T
    return np.dot(x_m, prin_comp)
