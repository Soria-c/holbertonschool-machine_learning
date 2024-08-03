#!/usr/bin/env python3
"""Maximization"""
import numpy as np


def maximization(X, g):
    """
    Calculates the maximization step in the EM algorithm for a GMM
    """
    if (not (isinstance(X, np.ndarray)) or X.ndim != 2) or\
       (not (isinstance(g, np.ndarray)) or g.ndim != 2):
        return None, None, None
    if g.shape[1] != X.shape[0]:
        return None, None, None
    if not np.isclose(np.sum(g, axis=0), [1])[0]:
        return None, None, None
    pi = []
    m = []
    S = []
    for k in range(len(g)):
        pi.append(np.sum(g[k] / len(X)))
        denm = np.sum(g[k])
        uk = np.sum(g[k][..., np.newaxis] * X, axis=0) / denm
        m.append(uk)
        diff = X-uk
        S.append(diff.T @ (g[k][..., np.newaxis] * diff) / denm)
    return np.array(pi), np.array(m), np.array(S)
