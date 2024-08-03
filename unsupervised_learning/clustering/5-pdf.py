#!/usr/bin/env python3
"""GMM PDF"""
import numpy as np


def pdf(X, m, S):
    """
    Calculates the probability density function of a Gaussian distribution
    """
    if not (isinstance(X, np.ndarray)) or X.ndim != 2:
        return None
    if not (isinstance(m, np.ndarray)) or m.ndim != 1:
        return None
    if not (isinstance(S, np.ndarray)) or S.ndim != 2:
        return None
    if (X.shape[1] != m.shape[0] or S.shape[0] != S.shape[1] or
            S.shape[0] != m.shape[0]):
        return None
    k = len(m)
    det_s = np.linalg.det(S)
    inv_s = np.linalg.inv(S)
    a = 1 / (np.sqrt(((2 * np.pi) ** k) * det_s))

    x_u = X - m
    b = -0.5 * np.sum(np.matmul(x_u, inv_s) * x_u, axis=1)
    return np.maximum(a * np.exp(b), 1e-300)
