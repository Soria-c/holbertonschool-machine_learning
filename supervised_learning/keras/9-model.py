#!/usr/bin/env python3
"""Save and Load Model"""
import tensorflow.keras as K


def save_model(network, filename):
    """
    Saves an entire mode
    """
    network.save(filename)


def load_model(filename):
    """
    Loads an entire model
    """
    return K.models.load_model(filename)
