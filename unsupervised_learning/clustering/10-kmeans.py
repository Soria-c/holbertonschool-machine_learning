#!/usr/bin/env python3
"""Hello, sklearn!"""
import sklearn.cluster as skcl


def kmeans(X, k):
    """
    Performs K-means on a dataset
    """
    km = skcl.KMeans(n_clusters=k).fit(X)
    return km.cluster_centers_, km.labels_
