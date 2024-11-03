#!/usr/bin/env python3
"""
This function plays an episode using the
trained Q-table in the FrozenLake environment.
"""
import numpy as np


def play(env, Q, max_steps=100):
    """
    Plays an episode using the trained Q-table, always
    exploiting the best action.

    Parameters:
    env (gym.Env): The FrozenLake environment instance.
    Q (numpy.ndarray): The trained Q-table.
    max_steps (int): The maximum number of steps in the episode.

    Returns:
    int: The total reward for the episode.
    list: A list of rendered board states at each step.
    """
    # Initialize the environment and get the starting state
    state = env.reset()[0]
    total_reward = 0
    rendered_outputs = []

    for step in range(max_steps):
        # Render and store the current state of the environment
        rendered_outputs.append(env.render())

        # Exploit the best action from the Q-table
        action = np.argmax(Q[state])

        # Take the action in the environment
        next_state, reward, done, _, _ = env.step(action)

        # Update total reward
        total_reward += reward
        state = next_state

        if done:
            break

    # Render and store the final state of the environment
    rendered_outputs.append(env.render())

    return total_reward, rendered_outputs
