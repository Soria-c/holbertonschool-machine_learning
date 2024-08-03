#!/usr/bin/env python3
"""GMM PDF"""
import numpy as np


def pdf(X, m, S):
    """
    Calculates the probability density function of a Gaussian distribution
    """
    k = len(m)
    det_s = np.linalg.det(S)
    inv_s = np.linalg.inv(S)
    a = 1 / (np.sqrt(((2 * np.pi) ** k) * det_s))

    x_u = X - m
    print(x_u.shape)
    print(x_u.T.shape)
    print(np.matmul(x_u, inv_s).shape)
    print(inv_s.shape)
    b = -0.5 * np.sum(np.matmul(x_u, inv_s) * x_u, axis=1)

    return np.maximum(a * np.exp(b), 1e-300)
