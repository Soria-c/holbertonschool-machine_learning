#!/usr/bin/env python3
"""Mini-Batch"""


def update_variables_momentum(alpha, beta1, var, grad, v):
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
    var: numpy.ndarray
        variable to be updated
    grad: numpy.ndarray
        gradient of var
    v: float
        previous first moment of var
    Return:
    -------
    updated variable and the new moment, respectively
    """
    vt = (beta1 * v) + ((1 - beta1) * grad)
    var = var - (alpha * vt)
    return var, vt
