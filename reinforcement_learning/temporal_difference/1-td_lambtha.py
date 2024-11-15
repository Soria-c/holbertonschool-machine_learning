#!/usr/bin/env python3
"""
TD(λ) algorithm for estimating state values.
"""

import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000,
               max_steps=100, alpha=0.1, gamma=0.99):
    """
    Performs the TD(λ) algorithm for estimating state values
    with eligibility traces.

    Parameters:
    - env: Environment instance with reset and step methods.
    - V (np.ndarray): Array containing the value estimates for each state.
    - policy (function): A function that takes in a state and returns the
                         next action.
    - lambtha (float): Eligibility trace factor (0 <= lambtha <= 1).
    - episodes (int): Total number of episodes to train over.
    - max_steps (int): Maximum number of steps per episode.
    - alpha (float): Learning rate.
    - gamma (float): Discount rate.

    Returns:
    - V (np.ndarray): Updated value estimates for each state.
    """

    for episode in range(episodes):
        # Initialize eligibility traces for each state as zero
        eligibility_traces = np.zeros_like(V)

        # Reset environment and get the initial state
        state = env.reset()[0]

        for step in range(max_steps):
            # Choose action according to the current policy
            action = policy(state)

            # Take the action, observe the next state and reward
            next_state, reward, done, trunc, _ = env.step(action)

            # Calculate the TD error
            td_error = reward + gamma * V[next_state] - V[state]

            # Update eligibility trace for the current state
            eligibility_traces[state] += 1

            # Update the value estimates and eligibility traces for all states
            V += alpha * td_error * eligibility_traces
            eligibility_traces *= gamma * lambtha

            # Move to the next state
            state = next_state

            if done:
                break

    return V
