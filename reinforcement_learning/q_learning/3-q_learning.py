#!/usr/bin/env python3
"""
This function performs Q-learning on a FrozenLake environment.
"""

import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Trains the agent using Q-learning.

    Parameters:
    env (gym.Env): The FrozenLake environment instance.
    Q (numpy.ndarray): The Q-table to update.
    episodes (int): Total number of episodes to train over.
    max_steps (int): Maximum number of steps per episode.
    alpha (float): The learning rate.
    gamma (float): The discount rate.
    epsilon (float): Initial epsilon value for exploration.
    min_epsilon (float): Minimum epsilon value after decay.
    epsilon_decay (float): Decay rate for epsilon between episodes.

    Returns:
    numpy.ndarray: The updated Q-table.
    list: A list containing the total reward per episode.
    """
    total_rewards = []

    for episode in range(episodes):
        state = env.reset()[0]  # Initialize the starting state
        episode_reward = 0

        for step in range(max_steps):
            # Select an action using epsilon-greedy policy
            action = epsilon_greedy(Q, state, epsilon)

            # Take the action in the environment
            next_state, reward, done, _, _ = env.step(action)

            # Update reward if falling into a hole
            if reward == 0 and done:
                reward = -1

            # Update Q-value using the Q-learning update rule
            old_value = Q[state, action]
            next_max = np.max(Q[next_state])

            Q[state, action] = (1 - alpha) * old_value + alpha *\
                               (reward + gamma * next_max)

            # Update cumulative reward for the episode
            episode_reward += reward

            # Move to the next state
            state = next_state

            if done:
                break

        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * (1 - epsilon_decay))

        # Store the total reward obtained in this episode
        total_rewards.append(episode_reward)

    return Q, total_rewards
