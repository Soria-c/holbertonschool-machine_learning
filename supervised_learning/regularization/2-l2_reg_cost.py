#!/usr/bin/env python3
"""L2 Regularization Cost"""


def l2_reg_cost(cost, model):
    """
    Calculates the cost of a neural network with L2 regularization
    ```
    Parameters:
    -----------
    -cost is a tensor containing the cost of the
     network without L2 regularization
    -model is a Keras model that includes layers with L2 regularization
    """
    return model.losses + cost
