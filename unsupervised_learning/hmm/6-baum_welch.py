#!/usr/bin/env python3
"""The Baum-Welch Algorithm"""


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


def EM(Observation, Transition, Emission, Initial):
    """
    Expectation Maximization algorithm
    """

    T = Observation.size
    M, N = Emission.shape
    _, F = forward(Observation, Emission, Transition, Initial)
    _, B = backward(Observation, Emission, Transition, Initial)

    Xi = np.zeros((T, M, M))

    for t in range(T):
        if t == T - 1:
            op = F[:, t].reshape(M, 1) * Transition
            Xi[t, :, :] = op.copy()
            break

        op = F[:, t].reshape(M, 1) * Transition * Emission[:, Observation[t+1]]
        op = op * B[:, t+1]
        Xi[t, :, :] = op.copy()

    Xi = Xi / Xi.sum(axis=(1, 2)).reshape(T, 1, 1)

    Transition = (Xi[:T-1, :, :].sum(axis=0) /
                  Xi[:T-1, :, :].sum(axis=(0, 2)).reshape(M, 1))

    for k in range(N):
        idxs = Observation[:T] == k
        Emission[:, k] = Xi[idxs, :, :].sum(axis=(0, 2))/Xi.sum(axis=(0, 2))

    Initial = Xi[0].sum(axis=0)

    return Transition, Emission, Initial.reshape(M, 1)


def baum_welch(Observation, Transition, Emission, Initial, iterations=1000):
    """
    Performs the Baum-Welch algorithm for finding locally optimal
    transition and emission probabilities for a Hidden Markov Model
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
    if type(iterations) is not int or iterations <= 0:
        return None, None

    for i in range(iterations):
        Transition, Emission, Initial = EM(
            Observation, Transition, Emission, Initial)

    return Transition, Emission
