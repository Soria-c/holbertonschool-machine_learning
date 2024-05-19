#!/usr/bin/env python3
"""Adam Update Variables"""


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    Updates a variable using the gradient descent
    with momentum optimization algorithm
    ```
    Parameters:
    -----------
    alpha: float
        the learning rate
    beta1: float
        weight used for the first moment
    beta2: float
        weight used for the second moment
    var: numpy.ndarray
        variable to be updated
    grad: numpy.ndarray
        gradient of var
    epsilon: float
        small number to avoid division by zero
    v: float
        previous first moment of var
    s: float
        previous second moment of var
    t: int
        time step used for bias correction
    Return:
    -------
    updated variable, the new first moment,
    and the new second moment, respectively
    """
    vd = (beta1 * v) + ((1 - beta1) * grad)
    sd = (beta2 * s) + ((1 - beta2) * grad ** 2)
    vd_corrected = vd / (1 - beta1 ** t)
    sd_corrected = sd / (1 - beta2 ** t)
    var = var - (alpha * (vd_corrected / ((sd_corrected ** 0.5) + epsilon)))
    return var, vd, sd
