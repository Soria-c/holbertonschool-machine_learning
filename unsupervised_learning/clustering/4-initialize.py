#!/usr/bin/env python3
"""Initialize GMM"""
import numpy as np


kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    Initializes variables for a Gaussian Mixture Model
    """
    if not(isinstance(X, np.ndarray)) or X.ndim != 2:
        return None, None, None
    n, d = X.shape
    pi = np.array([1 / k]).repeat(k)
    if ((k_m := kmeans(X, k))[0] is not None):
        m = k_m[0]
    else:
        return None, None, None
    S = np.identity(d)[np.newaxis, ...].repeat(k, axis=0)
    return pi, m, S
