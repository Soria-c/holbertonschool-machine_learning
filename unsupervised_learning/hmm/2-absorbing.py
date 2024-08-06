#!/usr/bin/env python3
"""Absorbing Chains"""
import numpy as np


def absorbing(P):
    """
    Determines if a markov chain is absorbing
    """
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return False
    if (np.any(P.sum(axis=1) != 1)):
        return False
    n = P.shape[0]
    absorbing = np.where(np.diag(P) == 1)[0]
    if len(absorbing) == 0:
        return False

    reachability = np.linalg.matrix_power(P, n)

    for i in range(n):
        if i not in absorbing:
            if not np.any(reachability[i, absorbing]):
                return False
    return True
