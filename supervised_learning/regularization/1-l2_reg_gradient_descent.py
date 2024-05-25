#!/usr/bin/env python3
"""Gradient Descent with L2 Regularization"""
import numpy as np


def dsoft(A, Y):
    """
    Calculates the derivate of cost function for softmax
    ```
    Parameters
    ----------
    Y: np.Array (1, m)
        contains the correct labels for the input data
    A: np.Array (1, m)
        containS the activated output of the neuron for each example
    """
    return A - Y


def dtanh(A):
    """
    Function to apply the derivative tanh activation function

    ```
    Parameters
    ----------
    x : np.Array
        Unactivated output of the neuron

    Returns
    -------
    np.Array
        Activated output applying derivative tanh
    """
    return 1 - (A**2)


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Calculates the cost of a neural network with L2 regularization
    Parameters:
    -----------
    Y: numpy.ndarray(classes, m)
        one-hot that contains the correct labels for the data
        -classes is the number of classes
        -m is the number of data points
    cost: double
        cost of the network without L2 regularization
    lambtha: double
        regularization parameter
    weights: dict[numpy.array]
        dictionary of the weights and biases of the neural network
    cache: dict
        dictionary of the outputs of each layer of the neural network
    L: int
        number of layers in the neural network
    Returns:
    -------
    cost of the network accounting for L2 regularization
    """
    prev_dz = None
    prev_w = None
    m = len(Y[0])
    for i in range(L, 0, -1):
        if i == L:
            dz = dsoft(cache[f"A{i}"], Y)
        else:
            dz = np.matmul(prev_w.transpose(), prev_dz) *\
                dtanh(cache[f"A{i}"])
        dw = (np.matmul(dz, cache[f"A{i - 1}"].transpose()) / m)\
            + l2_gradient_descent(lambtha, m, weights[f"W{i}"])
        db = dz.mean(axis=1, keepdims=True)
        prev_dz = dz
        prev_w = weights[f"W{i}"]
        weights[f"W{i}"] = weights[f"W{i}"] - (alpha * dw)
        weights[f"b{i}"] = weights[f"b{i}"] - (alpha * db)


def l2_gradient_descent(lambtha, m, W):
    """L2 gradient descent"""
    return (lambtha / m) * W
