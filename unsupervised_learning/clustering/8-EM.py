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
    if type(iterations) is not int or iterations <= 0:
        return None, None, None, None, None
    if type(tol) is not float or tol < 0:
        return None, None, None, None, None
    if type(verbose) is not bool:
        return None, None, None, None, None
    pi, m, S = initialize(X, k)
    if pi is None:
        return None, None, None, None, None
    for i in range(iterations):
        if ((E := expectation(X, pi, m, S))[0] is not None) and\
           ((M := maximization(X, E[0]))[0] is not None):
            g, l_l = E
            if verbose and not (i % 10):
                print(f"Log Likelihood after {i} iterations: {round(l_l, 5)}")
            if (i) and (l_l - likelihood <= tol):
                if verbose and (i % 10):
                    print(
                        f"Log Likelihood after {i} iterations: {round(l_l, 5)}"
                        )
                return pi, m, S, g, l_l
            pi, m, S = M
        else:
            return None, None, None, None, None
        likelihood = l_l
    g, l_l = expectation(X, pi, m, S)
    if verbose:
        print(f"Log Likelihood after {iterations} iterations: {round(l_l, 5)}")
    return pi, m, S, g, l_l
