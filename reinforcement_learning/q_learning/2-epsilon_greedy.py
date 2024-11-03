#!/usr/bin/env python3
"""
This function selects the next action using an epsilon-greedy policy.
"""

import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Selects the next action using the epsilon-greedy policy.

    Parameters:
    Q (numpy.ndarray): The Q-table.
    state (int): The current state.
    epsilon (float): The epsilon value for exploration vs. exploitation.

    Returns:
    int: The index of the next action.
    """
    # Determine if we should explore or exploit
    if np.random.uniform(0, 1) < epsilon:
        # Explore: choose a random action
        action = np.random.randint(Q.shape[1])
    else:
        # Exploit: choose the best action from the Q-table
        action = np.argmax(Q[state])

    return action
