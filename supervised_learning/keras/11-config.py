#!/usr/bin/env python3
"""Save and Load Configuration"""
import tensorflow.keras as K


def save_config(network, filename):
    """
    Saves a model’s configuration in JSON format
    """
    with open(filename, 'w') as f:
        f.write(network.to_json())


def load_config(filename):
    """
    Loads a model with a specific configuration
    """
    with open(filename, 'r') as f:
        return K.models.model_from_json(f.read())
