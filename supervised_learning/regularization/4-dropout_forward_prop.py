#!/usr/bin/env python3
"""Forward Propagation with Dropout"""
import numpy as np


def softmax(x):
    """
    Softmax activation function
    """
    return np.exp(x) / np.sum(np.exp(x), axis=0, keepdims=True)


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conducts forward propagation using Dropout
    ```
    Parameters:
    -----------
    X is a numpy.ndarray of shape (nx, m) containing
        the input data for the network
    nx is the number of input features
    m is the number of data points
    weights is a dictionary of the weights and biases of the neural network
    L the number of layers in the network
    keep_prob is the probability that a node will be kept
    All layers except the last should use the tanh activation function
    The last layer should use the softmax activation function
    Returns: a dictionary containing the outputs of each layer
        and the dropout mask used on each layer (see example for format)
    """
    cache = {}
    cache["A0"] = X
    for i in range(L):
        sum = np.matmul(weights[f"W{i +1}"],
                        cache[f"A{i}"]) + weights[f"b{i +1}"]
        if i == L - 1:
            cache[f"A{i+1}"] = softmax(sum)
        else:
            # Dropout mask
            cache[f"D{i+1}"] = np.random.binomial(n=1,
                                                  p=keep_prob,
                                                  size=sum.shape)
            tanh_act = np.tanh(sum) * cache['D' + str(i+1)]
            cache['A' + str(i+1)] = tanh_act / keep_prob
    return cache
