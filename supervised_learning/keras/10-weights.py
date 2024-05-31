#!/usr/bin/env python3
"""Save and Load Weights"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='keras'):
    """
    Saves a model’s weights
    """
    network.save_weights(filename + "." + save_format)


def load_weights(network, filename):
    """
    Loads a model’s weights
    """
    network.load_weights(filename)
