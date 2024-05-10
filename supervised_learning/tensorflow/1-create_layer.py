#!/usr/bin/env python3
"""Create placeholder"""
import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    """Function to create a tensorflow layer"""
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    y = tf.keras.layers.Dense(n,
                              activation=activation, name="layer",
                              kernel_initializer=init)
    return y(prev)
