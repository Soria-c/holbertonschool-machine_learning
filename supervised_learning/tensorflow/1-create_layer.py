#!/usr/bin/env python3
"""Create placeholder"""
import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    """Function to create a tensorflow layer"""
    y = tf.keras.layers.Dense(n, activation=activation, name="layer")
    tf.keras.initializers.VarianceScaling(mode='fan_avg')
    return y(prev)
