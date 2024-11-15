#!/usr/bin/env python3
"""
Train an agent using Monte-Carlo policy gradient in a given environment.
"""

import numpy as np
policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """
    Trains an agent using Monte-Carlo policy gradient.

    Parameters:
    - env: The initial environment.
    - nb_episodes: The number of episodes used for training.
    - alpha (float): The learning rate.
    - gamma (float): The discount factor.

    Returns:
    - scores (list): All scores for each episode (sum of rewards).
    """
    # Initialize random weights with the shape of (state_space, action_space)
    weight = np.random.rand(env.observation_space.shape[0], env.action_space.n)
    scores = []

    # Training loop
    for episode in range(nb_episodes):
        initial_state = env.reset()

        # Handle cases where env.reset() returns (state, info) tuple
        state = initial_state[0] if isinstance(initial_state, tuple)\
            else initial_state
        episode_gradient = []
        episode_rewards = []
        score = 0

        # Render the environment if show_result is True
        # and episode is a multiple of 1000
        if show_result and episode % 1000 == 0:
            env.render()

        # Generate an episode
        done = False
        while not done:
            state = state.reshape(1, -1)  # Reshape for single sample input
            action, gradient = policy_gradient(state, weight)
            next_state, reward, done, trunc, _ = env.step(action)

            # Record the gradients and rewards
            episode_gradient.append(gradient)
            episode_rewards.append(reward)
            score += reward

            # Move to the next state
            state = next_state

        # Discounted rewards calculation
        discounted_rewards = np.zeros_like(episode_rewards, dtype=float)
        cumulative = 0
        for t in reversed(range(len(episode_rewards))):
            cumulative = cumulative * gamma + episode_rewards[t]
            discounted_rewards[t] = cumulative

        # Update weights using Monte-Carlo policy gradient
        for t in range(len(episode_gradient)):
            weight += alpha * episode_gradient[t] * discounted_rewards[t]

        # Store the score and print the current episode's result
        scores.append(score)
        print(f"Episode: {episode + 1} Score: {score}")

    return scores
