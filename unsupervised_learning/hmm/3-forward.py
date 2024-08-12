#!/usr/bin/env python3
"""The Forward Algorithm"""


import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    Performs the forward algorithm for a hidden markov model
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

    n, _ = Emission.shape

    F = (Initial.T * Emission[:, Observation[0]]).T

    for t in range(1, Observation.shape[0]):
        r_t = F[:, t-1].T * Transition.T.reshape(n, 1, n)
        r_t = r_t * Emission[:, Observation[t]].reshape(n, 1, 1)
        F = np.concatenate((F, r_t.sum(-1)), axis=1)
    P = np.sum(F[:, -1])
    return P, F
