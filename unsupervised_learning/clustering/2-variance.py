#!/usr/bin/env python3
"""K-means Clustering"""
import numpy as np


def get_min_index(X, C):
    """
    Get indices of closests centroids
    """
    points, dims = X.shape
    x = X.reshape(points, 1, dims)
    r = np.repeat(C[np.newaxis, ...], points, axis=0)
    dist = np.linalg.norm(x - r, axis=2)
    return np.argmin(dist, axis=1)


def variance(X, C):
    """
    Calculates the total intra-cluster variance for a data set
    """
    return ((X-C[get_min_index(X, C)])**2).sum()
