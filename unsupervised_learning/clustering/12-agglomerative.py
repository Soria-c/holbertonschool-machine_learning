#!/usr/bin/env python3
"""Agglomerative"""
import scipy.cluster.hierarchy as skhr
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """
    Performs agglomerative clustering on a dataset
    """

    Z = skhr.linkage(X, 'ward')
    scatter = skhr.fcluster(Z, dist, 'distance')

    skhr.dendrogram(Z, color_threshold=dist, above_threshold_color='b')
    plt.show()
    plt.close()
    return scatter
