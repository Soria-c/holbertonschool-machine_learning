#!/usr/bin/env python3
"""K-means Clustering"""
import numpy as np


def variance(X, C):
    """
    Calculates the total intra-cluster variance for a data set
    """
    # ric_c = np.repeat(C, X.shape[0], axis=0)
    points, dims = X.shape
    x = X.reshape(points, 1, dims)
    
    print(X.shape)
    # print(x[:5])
    # print(X[:5])
    r = np.repeat(C[np.newaxis, ...], points, axis=0)
    print(r[:1])
    print(C)
    print(r.shape)
    print(x.shape)
    print(C.shape)
    print(x[:1])
    x = x - r
    # print(x.shape)
    # print(x[:1])
    dist = np.linalg.norm(x, axis=2)
    print(dist.shape)
    print(dist[:1])

    mins = np.argmin(dist, axis=1)
    print(mins)
