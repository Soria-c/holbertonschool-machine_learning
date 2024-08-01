#!/usr/bin/env python3
"""Optimize k"""
import numpy as np


kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Tests for the optimum number of clusters by variance
    """
    if type(X) is not np.ndarray or X.ndim != 2:
        return None, None
    if type(kmin) is not int or kmin <= 0:
        return None, None
    if kmax is None:
        kmax = X.shape[0]
    if type(kmax) is not int or kmax <= 0:
        return None, None
    if kmax <= kmin:
        return None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None
    results, d_vars = [], []
    for i in range(kmin, kmax+1):
        centroids, indices = kmeans(X, i, iterations)
        if i == kmin:
            current_variance = variance(X, centroids)
            results.append((centroids, indices))
            d_vars.append(0.0)
            continue
        variance_diff = abs(current_variance-variance(X, centroids))
        results.append((centroids, indices))
        d_vars.append(variance_diff)
    return results, d_vars
