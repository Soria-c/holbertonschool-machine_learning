#!/usr/bin/env python3
"""Principal Components Analysis"""
import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on a dataset
    """
    U, S, V = np.linalg.svd(X)

    variance = S**2 / (len(X) - 1)

    total_variance = np.sum(variance)

    explained_variance_ratio = variance / total_variance

    cumulative_variance = np.cumsum(explained_variance_ratio)
    valid_comp = np.argmax(cumulative_variance >= var) + 1

    return V[:valid_comp + 1].T
