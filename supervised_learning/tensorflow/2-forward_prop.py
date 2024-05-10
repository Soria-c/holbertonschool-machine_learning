#!/usr/bin/env python3
"""Create placeholder"""
import tensorflow.compat.v1 as tf


create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Function that creates the forward propagation graph for the neural network
    ```
    Parameters:
    -----------
        x is the placeholder for the input data
        layer_sizes is a list containing the number of nodes in each
            layer of the network
        activations is a list containing the activation
            functions for each layer of the network
    """
    for i, layer_size in enumerate(layer_sizes):
        if not i:
            prev_layer = create_layer(x, layer_size, activations[i])
        else:
            prev_layer = create_layer(prev_layer, layer_size, activations[i])
    return prev_layer
