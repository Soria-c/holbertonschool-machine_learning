#!/usr/bin/env python3
"""Input"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network with the Keras library
    ```
    Parameters:
    -----------
    nx: int
        number of input features to the network
    layers: int
        is a list containing the number of nodes in each layer of the network
    activations: list
        list containing the activation functions used
        for each layer of the network
    lambtha:
        L2 regularization parameter
    keep_prob:
        probability that a node will be kept for dropout
    """
    n_layers = len(layers)
    inputs = K.Input(shape=(nx,))
    outputs = None
    for i in range(n_layers):
        if not i:
            layer_recur = K.layers.Dense(
                layers[i],
                activation=activations[i],
                kernel_regularizer=K.regularizers.L2(lambtha)
                )(inputs)
        else:
            layer_recur = K.layers.Dense(
                layers[i],
                activation=activations[i],
                kernel_regularizer=K.regularizers.L2(lambtha)
                )(layer_recur)
            outputs = layer_recur
        if n_layers > 1 and (i != n_layers - 1):
            layer_recur = K.layers.Dropout(1 - keep_prob)(layer_recur)
    return K.Model(inputs=inputs, outputs=outputs)
