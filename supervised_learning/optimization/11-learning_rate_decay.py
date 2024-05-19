#!/usr/bin/env python3
"""Learning Rate Decay"""


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Updates the learning rate using inverse time decay
    ```
    Parameters:
    -----------
    alpha: float
        the learning rate
    decay_rate: float
        weight used to determine the rate at which alpha will decay
    global_step: float
        number of passes of gradient descent that have elapsed
    decay_step: float
        number of passes of gradient descent that
        should occur before alpha is decayed further
    Return:
    -------
    updated value for alpha
    """
    return (1 / (1 + decay_rate * (int(global_step / decay_step)))) * alpha
