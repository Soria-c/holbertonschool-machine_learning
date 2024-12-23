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
        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros(shape=(nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    def sig(self, x):
        """
        Function to apply the sigmoid activation function

        ```
        Parameters
        ----------
        x : np.Array
            Unactivated output of the neuron

        Returns
        -------
        np.Array
            Activated output applying sigmoid
        """
        return 1/(1 + np.exp(-x))

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network
        ```
        Parameters
        ----------
        X : np.Array
            Contains the input data
        """
        self.__A1 = self.sig(np.matmul(self.__W1, X) + self.__b1)
        self.__A2 = self.sig(np.matmul(self.__W2, self.__A1) + self.__b2)
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        ```
        Parameters
        ----------
        Y: np.Array (1, m)
            contains the correct labels for the input data
        A: np.Array (1, m)
            containS the activated output of the neuron for each example
        """
        return np.mean(-((Y * np.log(A)) + (1 - Y)*(np.log(1.0000001 - A))))

    def evaluate(self, X, Y):
        """
        Evaluates the neuron’s predictions
        ```
        Parameters
        ----------
        X: np.Array (nx, m)
            contains input data
        Y: np.Array (1, m)
            contains the correct labels for the input data
        """
        self.forward_prop(X)
        return ((self.__A2 >= 0.50).astype(int), self.cost(Y, self.__A2))

    @property
    def W1(self):
        """
        Getter of __W1
        """
        return self.__W1

    @property
    def b1(self):
        """
        Getter of __b1
        """
        return self.__b1

    @property
    def A1(self):
        """
        Getter of __A1
        """
        return self.__A1

    @property
    def W2(self):
        """
        Getter of __W2
        """
        return self.__W2

    @property
    def b2(self):
        """
        Getter of __b2
        """
        return self.__b2

    @property
    def A2(self):
        """
        Getter of __A2
        """
        return self.__A2
