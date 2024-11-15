#!/usr/bin/env python3
"""
Train an agent using Monte-Carlo policy gradient in a given environment.
"""

import numpy as np

policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98):
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
    scores = []
    weight = np.random.rand(4, 2)  # Initialize weights for the policy gradient

    for episode in range(nb_episodes):
        state, _ = env.reset()
        done = False
        episode_rewards = []
        gradients = []

        while not done:
            # Render environment every 1000 episodes if show_result is True
            if show_result and episode % 1000 == 0:
                env.render()

            # Compute action and gradient using policy gradient
            # Use correct policy_gradient function
            action, grad = policy_gradient(state, weight)
            action = int(action)  # Ensure action is an integer

            # Take a step in the environment
            next_state, reward, done, _, _ = env.step(action)

            # Collect reward and gradient for this step
            episode_rewards.append(reward)
            gradients.append(grad)
            state = next_state

        # Compute the score (sum of rewards for the episode)
        score = sum(episode_rewards)
        scores.append(score)

        # Print the episode number and score
        print(f"Episode: {episode} Score: {score}")

        # Compute the discounted rewards and update weights
        for t in range(len(episode_rewards)):
            G = sum([gamma ** i * episode_rewards[i + t]
                    for i in range(len(episode_rewards) - t)])

            # Update weights using policy gradient
            weight_update = alpha * G * gradients[t]
            weight += weight_update

    return scores
