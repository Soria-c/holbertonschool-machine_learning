#!/usr/bin/env python3
"""Markov Chain"""
import numpy as np


def regular(P):
    """
    Determines the probability of a markov chain
    being in a particular state after a specified number of iterations
    """
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return None
    if (np.any(P.sum(axis=1) != 1)):
        return None

    e_vals, e_vecs = np.linalg.eig(P.T)
    e_vec1 = e_vecs[:, np.isclose(e_vals, 1)]
    if (len(e_vec1[0]) != 1):
        return None
    e_vec1 = e_vec1[:, 0]
    stationary = e_vec1 / e_vec1.sum()
    return stationary.real
