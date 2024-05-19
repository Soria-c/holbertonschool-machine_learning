#!/usr/bin/env python3
"""RMSProp"""


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Updates a variable using the gradient descent
    with momentum optimization algorithm
    ```
    Parameters:
    -----------
    alpha: float
        the learning rate
    beta2: float
        RMSProp weight
    var: numpy.ndarray
        variable to be updated
    grad: numpy.ndarray
        gradient of var
    epsilon: float
        small number to avoid division by zero
    s: float
        previous first moment of var
    Return:
    -------
    updated variable and the new moment, respectively
    """
    s0 = (beta2 * s) + ((1 - beta2) * grad ** 2)
    var = var - (alpha * (grad / ((s0 ** 0.5) + epsilon)))
    return var, s0
