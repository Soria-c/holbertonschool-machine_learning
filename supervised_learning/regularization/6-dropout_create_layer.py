#!/usr/bin/env python3
"""Create a Layer with Dropout"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """
    Creates a layer of a neural network using dropout
    ```
    Parameters:
    -----------
    prev is a tensor containing the output of the previous layer
    n is the number of nodes the new layer should contain
    activation is the activation function for the new layer
    keep_prob is the probability that a node will be kept
    training is a boolean indicating whether the model is in training mode
    Returns: the output of the new layer
    """
    weights = tf.keras.initializers.VarianceScaling(scale=2.0,
                                                    mode='fan_avg')
    layer = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(units=n,
                                  activation=activation,
                                  kernel_initializer=weights),
            tf.keras.layers.Dropout(rate=keep_prob*n),
        ]
    )
    return layer(inputs=prev, training=training)