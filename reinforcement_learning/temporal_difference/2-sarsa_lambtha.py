#!/usr/bin/env python3
"""
SARSA(λ) algorithm for updating Q-values with eligibility traces.
"""

import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Performs the SARSA(λ) algorithm for estimating Q-values
    with eligibility traces.

    Parameters:
    - env: Environment instance with reset and step methods.
    - Q (np.ndarray): Array containing the Q-value estimates
      for each state-action pair.
    - lambtha (float): Eligibility trace factor (0 <= lambtha <= 1).
    - episodes (int): Total number of episodes to train over.
    - max_steps (int): Maximum number of steps per episode.
    - alpha (float): Learning rate.
    - gamma (float): Discount rate.
    - epsilon (float): Initial threshold for epsilon-greedy action selection.
    - min_epsilon (float): Minimum value that epsilon should decay to.
    - epsilon_decay (float): Decay rate for updating epsilon between episodes.

    Returns:
    - Q (np.ndarray): Updated Q-value estimates for each state-action pair.
    """

    def epsilon_greedy_action(Q, state, epsilon):
        """Selects an action using epsilon-greedy policy based on Q-table."""
        if np.random.uniform(0, 1) > epsilon:
            return np.argmax(Q[state, :])
        return np.random.randint(0, Q.shape[1])

    for episode in range(episodes):
        # Initialize eligibility traces for each state-action pair as zero
        eligibility_traces = np.zeros_like(Q)

        # Reset environment and initialize the first state and action
        state = env.reset()[0]
        action = epsilon_greedy_action(Q, state, epsilon)

        for step in range(max_steps):
            # Take action, observe next state, reward, and done flag
            next_state, reward, done, trunc, _ = env.step(action)
            next_action = epsilon_greedy_action(Q, next_state, epsilon)

            # Calculate TD error for SARSA
            td_error = reward + gamma * Q[next_state, next_action]\
                - Q[state, action]

            # Update eligibility trace for the current state-action pair
            eligibility_traces[state, action] += 1

            # Update Q-values and eligibility traces for all state-action pairs
            eligibility_traces *= gamma * lambtha
            Q += alpha * td_error * eligibility_traces

            # Move to the next state and action
            state, action = next_state, next_action

            # End episode if done
            if done or trunc:
                break

        # Decay epsilon after each episode
        epsilon = (min_epsilon + (epsilon - min_epsilon) *
                   np.exp(-epsilon_decay * episode))

    return Q
