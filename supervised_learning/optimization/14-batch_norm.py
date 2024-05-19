#!/usr/bin/env python3
"""Learning Rate Decay"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network in tensorflow
    ```
    Parameters:
    -----------
    prev:
        is the activated output of the previous layer
    n:
        is the number of nodes in the layer to be created
    activation:
        is the activation function that should
        be used on the output of the layer
    Return:
    -------
    tensor of the activated output for the layer
    """
    weights = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.keras.layers.Dense(n, kernel_initializer=weights)
    z = layer(prev)
    mean, variance = tf.nn.moments(z, 0)
    gamma = tf.Variable(tf.ones(n), trainable=True)
    beta = tf.Variable(tf.zeros(n), trainable=True)
    epsilon = 1e-7
    batch_norm = tf.nn.batch_normalization(
        x=z,
        mean=mean,
        variance=variance,
        offset=beta,
        scale=gamma,
        variance_epsilon=epsilon
    )
    return activation(batch_norm)
