#!/usr/bin/env python3
"""Markov Chain"""
import numpy as np


def markov_chain(P, s, t=1):
    """
    Determines the probability of a markov chain
    being in a particular state after a specified number of iterations
    """
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return None
    if not isinstance(s, np.ndarray) or s.ndim != 2:
        return None
    if (P.shape[0] != s.shape[1]):
        return None
    if t < 1:
        return None

    prob_dis = s
    for i in range(t):
        prob_dis = prob_dis @ P
    return prob_dis
