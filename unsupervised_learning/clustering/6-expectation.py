#!/usr/bin/env python3
"""Expectation"""
import numpy as np


pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Calculates the expectation
    step in the EM algorithm for a GMM
    """
    if not (isinstance(pi, np.ndarray)) or pi.ndim != 1:
        return None, None
    if not (isinstance(S, np.ndarray)) or S.ndim != 3:
        return None, None
    if (pi.shape[0] != S.shape[0]):
        return None, None
    if not np.isclose([np.sum(pi)], [1])[0]:
        return None, None

    probs = []
    for i, v in enumerate(S):
        if (gmm_pdf := pdf(X, m[i], v)) is not None:
            probs.append(gmm_pdf)
        else:
            return None, None
    probs = np.array(probs)
    pi = pi[..., np.newaxis]
    s = pi * probs
    denm = np.sum(s, axis=0)
    g = s / denm
    log_l = np.sum(np.log(denm))
    return g, log_l
