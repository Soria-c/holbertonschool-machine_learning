#!/usr/bin/env python3
import numpy as np
"""L2 Regularization Cost"""


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates the cost of a neural network with L2 regularization
    Parameters:
    -----------
    cost: double
        cost of the network without L2 regularization
    lambtha: double
        regularization parameter
    weights: dict[numpy.array]
        dictionary of the weights and biases of the neural network
    L: int
        number of layers in the neural network
    m: int
        number of data points used
    Returns:
    -------
    cost of the network accounting for L2 regularization
    """
    norm = 0
    for i in weights.values():
        norm += np.linalg.norm(i) ** 2
    decay_w = (lambtha / (2 * m)) * norm
    return cost + decay_w
