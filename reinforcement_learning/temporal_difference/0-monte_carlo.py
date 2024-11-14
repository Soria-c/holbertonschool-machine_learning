#!/usr/bin/env python3
"""
Monte Carlo algorithm for estimating state values.
"""

import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """
    Performs the Monte Carlo algorithm for estimating state values.

    Parameters:
    - env: Environment instance with reset and step methods.
    - V (np.ndarray): Array containing the value estimates for each state.
    - policy (function): A function that takes in a state
      and returns the next action.
    - episodes (int): Total number of episodes to train over.
    - max_steps (int): Maximum number of steps per episode.
    - alpha (float): Learning rate.
    - gamma (float): Discount rate.

    Returns:
    - V (np.ndarray): Updated value estimates for each state.
    """

    for episode in range(episodes):
        # Generate an episode
        state = env.reset()
        episode_data = []

        for step in range(max_steps):
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            episode_data.append((state, reward))

            if done:
                break
            state = next_state

        # Compute returns for each state in the episode
        G = 0  # Initialize return
        visited_states = set()

        # Reverse the episode to calculate returns for each state
        for state, reward in reversed(episode_data):
            G = reward + gamma * G

            # Only update if it's the first time the state
            # is visited in the episode
            if state not in visited_states:
                visited_states.add(state)

                # Update value estimate using incremental mean
                V[state] = V[state] + alpha * (G - V[state])

    return V
