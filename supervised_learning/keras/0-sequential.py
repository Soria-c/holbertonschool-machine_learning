#!/usr/bin/env python3
"""Sequential"""
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
    model = K.Sequential()
    for i in range(len(layers)):
        if not i:
            model.add(K.layers.Dense(
                layers[i],
                activation=activations[i],
                input_shape=(nx,),
                kernel_regularizer=K.regularizers.L2(lambtha)
                ))
        else:
            model.add(K.layers.Dense(
                layers[i],
                activation=activations[i],
                kernel_regularizer=K.regularizers.L2(lambtha)
                ))
        if i != len(layers) - 1:
            model.add(K.layers.Dropout(keep_prob))
    return model
