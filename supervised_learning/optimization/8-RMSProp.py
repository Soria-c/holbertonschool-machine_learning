#!/usr/bin/env python3
"""RMSProp Upgraded"""
import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """
    Sets up the RMSProp optimization algorithm in TensorFlow
    ```
    Parameters:
    -----------
    alpha: float
        the learning rate
    beta2: float
        the momentum weight
    epsilon: float
        is a small number to avoid division by zero
    Return:
    -------
    optimizer
    """
    return tf.keras.optimizers.RMSprop(
        learning_rate=alpha, rho=beta2, epsilon=epsilon)
