#!/usr/bin/env python3
"""This module defines a representation of a Neural Network"""
import numpy as np


class NeuralNetwork:
    """
    Defines a neural network with one hidden layer performing binary
    classification
    ```
    Attributes
    ----------
    W1 : np.Array[Float](nodes, nx)
        The weights vector for the hidden layer
    b1 : np.Array[Float](1, nodes)
        The bias for the hidden layer.
    A1 : float
        The activated output for the hidden layer.
    W2 : np.Array[Float]
        The weights vector for the output neuron.
    b2 : float
        The bias for the output neuron.
    A2 : float
        The activated output for the output neuron (prediction).
    """

    def __init__(self, nx, nodes):
        """
        Constructor

        ```
        Parameters
        ----------
        nx: int
            is the number of input features to the neuron
        nodes: int
            number of nodes found in the hidden layer
        """
        if (not isinstance(nx, int)):
            raise TypeError("nx must be an integer")
        if (nx < 1):
            raise ValueError("nx must be a positive integer")
        if (not isinstance(nodes, int)):
            raise TypeError("nodes must be an integer")
        if (nodes < 1):
            raise ValueError("nodes must be a positive integer")
        self.W1 = np.random.normal(size=(nodes, nx))
        self.b1 = np.zeros(shape=(nodes, 1))
        self.A1 = 0
        self.W2 = np.random.normal(size=(1, nodes))
        self.b2 = 0
        self.A2 = 0
