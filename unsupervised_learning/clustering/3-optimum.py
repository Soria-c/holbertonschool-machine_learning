#!/usr/bin/env python3
"""Optimize k"""
import numpy as np


kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Tests for the optimum number of clusters by variance
    """
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
