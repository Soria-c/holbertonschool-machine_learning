#!/usr/bin/env python3
"""Initialize K-means"""
import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means
    """
    return np.random.uniform(
        np.min(X, axis=0), np.max(X, axis=0), size=(k, X.shape[1]))
