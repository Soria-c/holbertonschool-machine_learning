#!/usr/bin/env python3
"""
Policy function to compute action probabilities from a state
matrix and weight matrix.
"""

import numpy as np


def policy(matrix, weight):
    """
    Computes the policy by applying weights to the input state matrix.

    Parameters:
    - matrix (np.ndarray): The state matrix, with shape (n, m) where
                           n is the number of samples and m is the
                           state dimension.
    - weight (np.ndarray): The weight matrix, with shape (m, k) where
                           k is the number of possible actions.

    Returns:
    - action_probs (np.ndarray): The probability distribution over actions
                                 for each sample, with shape (n, k).
    """
    # Linear transformation of states by weights
    z = np.dot(matrix, weight)

    # Apply softmax to each row to get probabilities
    # Stability adjustment
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    action_probs = exp_z / np.sum(exp_z, axis=1, keepdims=True)

    return action_probs
