#!/usr/bin/env python3
"""The Viretbi Algorithm"""


import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """
    Calculates the most likely sequence of hidden states for a hidden
    markov model
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

    N, _ = Emission.shape
    T = Observation.size

    s_prob = Initial * Emission[:, Observation[0]][..., np.newaxis]
    buffer = np.zeros((N, T))

    for t in range(1, T):
        mat = (Emission[:, Observation[t]] * Transition.reshape(N, 1, N))
        mat = (mat.reshape(N, N) * s_prob[:, t-1].reshape(N, 1))

        mx = np.max(mat, axis=0).reshape(N, 1)
        s_prob = np.concatenate((s_prob, mx), axis=1)
        buffer[:, t] = np.argmax(mat, axis=0).T

    P = np.max(s_prob[:, T-1])
    link = np.argmax(s_prob[:, T-1])
    path = [link]

    for t in range(T - 1, 0, -1):
        idx = int(buffer[link, t])
        path.append(idx)
        link = idx

    return path[::-1], P
