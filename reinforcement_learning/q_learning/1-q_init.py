#!/usr/bin/env python3
"""
This function initializes the Q-table for a FrozenLake environment.
"""

import numpy as np


def q_init(env):
    """
    Initializes the Q-table for the given FrozenLake environment.

    Parameters:
    env (gym.Env): The FrozenLake environment instance.

    Returns:
    numpy.ndarray: A Q-table initialized to zeros with shape
        (num_states, num_actions).
    """
    # Get the number of states and actions from the environment
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    # Initialize the Q-table as a 2D array of zeros
    Q_table = np.zeros((num_states, num_actions))

    return Q_table
