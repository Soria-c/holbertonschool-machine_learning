#!/usr/bin/env python3
"""Create Momemtum Op"""
import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """
    Updates a variable using the gradient descent
    with momentum optimization algorithm
    ```
    Parameters:
    -----------
    alpha: float
        the learning rate
    beta1: float
        the momentum weight
    Return:
    -------
    optimizer
    """
    return tf.train.MomentumOptimizer(alpha, beta1)
