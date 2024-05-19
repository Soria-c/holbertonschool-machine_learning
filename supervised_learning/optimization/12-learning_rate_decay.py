#!/usr/bin/env python3
"""Learning Rate Decay"""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """
    Updates the learning rate using inverse time decay
    ```
    Parameters:
    -----------
    alpha: float
        the learning rate
    decay_rate: float
        weight used to determine the rate at which alpha will decay
    decay_step: float
        number of passes of gradient descent that
        should occur before alpha is decayed further
    Return:
    -------
    learning rate decay operation
    """
    return tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=alpha,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True
    )
