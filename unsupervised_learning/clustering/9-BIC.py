#!/usr/bin/env python3
"""Maximization"""
import numpy as np


expectation_maximization = __import__('8-EM').expectation_maximization


def __bic(p, n, likelihood):
    """
    Bayesian Information Criterion
    """
    return p * np.log(n) - 2 * likelihood


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Finds the best number of clusters for a GMM
    using the Bayesian Information Criterion
    """

    pi, m, S, g, l_log = expectation_maximization(
        X, kmin, iterations, tol, verbose)
    if pi is None:
        return None, None, None, None
    n, d = X.shape
    best_i = 0
    ks = [kmin]
    results = [(pi, m, S)]
    likelihoods = [l_log]
    bics = [__bic(1+(kmin*d)+(kmin*d), n, l_log)]

    kmax = n if kmax is None else kmax

    for i, ki in enumerate(range(kmin + 1, kmax + 1), 1):
        pi, m, S, g, l_log = expectation_maximization(
            X, ki, iterations, tol, verbose)
        bic = __bic(1+(ki*d)+(ki*d), n, l_log)
        ks.append(ki)
        results.append((pi, m, S))
        likelihoods.append(l_log)
        bics.append(bic)
        if bics[best_i] >= bic:
            best_i = i
    return ks[best_i], results[best_i], likelihoods, bics
