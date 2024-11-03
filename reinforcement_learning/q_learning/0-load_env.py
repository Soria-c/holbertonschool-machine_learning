#!/usr/bin/env python3
"""
This function loads the FrozenLake environment from gymnasium with optional
customization for map description, map name, and slipperiness.
"""

import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    Loads the FrozenLake environment from gymnasium.

    Parameters:
    desc (list of lists, optional): Custom description of the map.
    map_name (str, optional): Pre-made map name to load (e.g., "4x4" or "8x8").
    is_slippery (bool): If True, the ice is slippery; otherwise, it is not.

    Returns:
    gym.Env: The loaded FrozenLake environment.
    """
    # Load the environment with specified settings
    env = gym.make(
        "FrozenLake-v1",
        desc=desc,
        map_name=map_name,
        is_slippery=is_slippery
    )
    return env
