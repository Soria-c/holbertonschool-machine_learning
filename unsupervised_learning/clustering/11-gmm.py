#!/usr/bin/env python3
"""SK GMM"""
import sklearn.mixture as skmx


def gmm(X, k):
    """
    Calculates a GMM from a dataset:
    """
    gm = skmx.GaussianMixture(n_components=k).fit(X)
    return gm.weights_, gm.means_, \
        gm.covariances_, gm.predict(X), \
        gm.bic(X)
