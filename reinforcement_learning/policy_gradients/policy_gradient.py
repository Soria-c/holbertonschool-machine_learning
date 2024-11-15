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


def policy_gradient(state, weight):
    """
    Computes the Monte-Carlo policy gradient based
    on a state and weight matrix.

    Parameters:
    - state (np.ndarray): Matrix representing the current
                          observation of the environment.
    - weight (np.ndarray): Matrix of random weights.

    Returns:
    - action (int): The chosen action based on the policy.
    - gradient (np.ndarray): The gradient of the log-probability
                             of the chosen action.
    """
    # Calculate the action probabilities using the policy function
    action_probs = policy(state, weight)

    # Sample an action based on the probability distribution
    action = np.random.choice(len(action_probs[0]), p=action_probs[0])

    # Calculate the gradient of the log-probability for the chosen action
    dsoftmax = action_probs.copy()
    # Subtract 1 from the probability of the chosen action
    dsoftmax[0, action] -= 1

    # Compute the gradient with respect to the weights
    gradient = np.dot(state.T, dsoftmax)

    return action, gradient
