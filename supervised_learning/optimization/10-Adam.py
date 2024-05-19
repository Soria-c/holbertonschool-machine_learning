#!/usr/bin/env python3
"""Adam Upgraded"""
import tensorflow as tf


def create_Adam_op(alpha, beta1, beta2, epsilon):
    """
    Sets up the RMSProp optimization algorithm in TensorFlow
    ```
    Parameters:
    -----------
    alpha: float
        the learning rate
    beta1: float
        weight used for the first moment
    beta2: float
        weight used for the second moment
    epsilon: float
        is a small number to avoid division by zero
    Return:
    -------
    optimizer
    """
    return tf.keras.optimizers.Adam(
        learning_rate=alpha, beta_1=beta1, beta_2=beta2, epsilon=epsilon)
