#!/usr/bin/env python3
"""The Backward Algorithm"""


import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """
    Performs the backward algorithm for a hidden markov model
    """

    if type(Observation) is not np.ndarray or Observation.ndim != 1:
        return None, None
    if Observation.shape[0] == 0:
        return None, None
    if type(Emission) is not np.ndarray or Emission.ndim != 2:
        return None, None
    if type(Transition) is not np.ndarray or Transition.ndim != 2:
        return None, None
    if type(Initial) is not np.ndarray or len(Initial) != Transition.shape[0]:
        return None, None

    N, M = Emission.shape
    T = Observation.size

    B = np.ones((N, T), dtype="float")

    for t in range(T-2, -1, -1):
        mat = (Emission[:, Observation[t+1]] * Transition.reshape(N, 1, N))
        mat = (B[:, t+1] * mat).reshape(N, N).sum(axis=1)

        B[:, t] = mat

    P = (Initial.T * Emission[:, Observation[0]] * B[:, 0])

    return P.sum(axis=1)[0], B
