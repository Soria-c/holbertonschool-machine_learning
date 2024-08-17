#!/usr/bin/env python3
"""K-means Clustering"""
import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means
    """
    if k == 0:
        return
    try:
        return np.random.uniform(
            np.min(X, axis=0), np.max(X, axis=0), size=(k, X.shape[1]))
    except Exception as e:
        return


def get_min_index(X, C):
    """
    Get indices of closests centroids
    """
    points, dims = X.shape
    x = X.reshape(points, 1, dims)
    r = np.repeat(C[np.newaxis, ...], points, axis=0)
    dist = np.linalg.norm(x - r, axis=2)
    return np.argmin(dist, axis=1)


def kmeans(X, k, iterations=1000):
    """
    Performs K-means on a dataset
    """
    if type(iterations) is not int or iterations <= 0:
        return None, None
    ic_centroids = initialize(X, k)
    if ic_centroids is None:
        return None, None
    for _ in range(iterations):
        ric_c = np.repeat(ic_centroids, X.shape[0], axis=0)
        b = np.vsplit(ric_c, k)
        d = []
        for i in range(k):
            dis = np.linalg.norm(X-b[i], axis=1)
            d.append(dis)
        means = np.zeros(ic_centroids.shape)
        index = np.array(d).argmin(axis=0)
        sort_index_i = np.unique(np.sort(index), return_index=True)
        s = np.split(X[np.argsort(index)], sort_index_i[1])

        means[sort_index_i[0]] = np.array(list(map(
            lambda x: np.mean(x, axis=0), s[1:])))

        miss = np.setdiff1d(np.arange(k), sort_index_i[0])
        if (len(miss) > 0):
            re_init = initialize(X, 1)
            means[miss] = re_init[0]
        if (np.array_equal(ic_centroids, means)):
            return ic_centroids, index
        ic_centroids = means
    return ic_centroids, get_min_index(X, ic_centroids)
