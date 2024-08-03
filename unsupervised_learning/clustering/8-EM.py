#!/usr/bin/env python3
"""Maximization"""
import numpy as np


initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Performs the expectation maximization for a GMM
    """
    pi, m, S = initialize(X, k)
    if pi is None:
        return None, None, None, None, None
    for i in range(iterations):
        if ((E := expectation(X, pi, m, S))[0] is not None) and\
           ((M := maximization(X, E[0]))[0] is not None):
            g, log_l = E
            pi, m, S = M
            if verbose and not (i % 10):
                print(f"Log Likelihood after {i} iterations: {log_l:.5f}")
        else:
            return None, None, None, None, None
        if (i) and (log_l - likelihood <= tol):
            if verbose and (i % 10):
                print(f"Log Likelihood after {i} iterations: {log_l:.5f}")
            return ppi, pm, pS, g, log_l
        likelihood = log_l
        pm, ppi, pS = m, pi, S
    if verbose and not (i % 10):
        print(f"Log Likelihood after {iterations} iterations: {log_l:.5f}")
    return pi, m, S, g, log_l
