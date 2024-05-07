#!/usr/bin/env python3
"""This module defines a representation of a Neural Network"""
import numpy as np
import matplotlib.pyplot as plt


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

    def dzig(self, x):
        """
        Function to apply the derivative sigmoid activation function

        ```
        Parameters
        ----------
        x : np.Array
            Unactivated output of the neuron

        Returns
        -------
        np.Array
            Activated output applying derivative sigmoid
        """
        return x * (1 - x)

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

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neuron
        ```
        Parameters
        ----------
        X: np.Array (nx, m)
            contains input data
        Y: np.Array (1, m)
            contains the correct labels for the input data
        A1: np.Array
            is the output of the hidden layer
        A2: np.Array
            is the predicted output
        alpha: float
            learning rate
        """
        dz2 = A2 - Y
        dw2 = np.matmul(dz2, A1.transpose()) / len(A2[0])
        db2 = dz2.mean(axis=1, keepdims=True)

        dz1 = np.matmul(self.__W2.transpose(), dz2) * self.dzig(A1)
        dw1 = np.matmul(dz1, X.transpose()) / len(A2[0])
        db1 = dz1.mean(axis=1, keepdims=True)

        self.__W1 = self.__W1 - (alpha * dw1)
        self.__b1 = self.__b1 - (alpha * db1)

        self.__W2 = self.__W2 - (alpha * dw2)
        self.__b2 = self.__b2 - (alpha * db2)

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """
        Trains the neuron
        ```
        Parameters
        ----------
        X: np.Array (nx, m)
            contains input data
        Y: np.Array (1, m)
            contains the correct labels for the input data
        A: np.Array (1, m)
            containS the activated output of the neuron for each example
        alpha: float
            learning rate
        iterations: int
            number of iterations to train over
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        epochs = []
        costs = []
        for i in range(iterations):
            if verbose and (i % step) == 0:
                current_cost = self.cost(Y, self.__A)
                costs.append(current_cost)
                epochs.append(i)
                print(f"Cost after {i} iterations: {current_cost}")
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A1, self.__A2, alpha)
        if graph is True:
            plt.plot(epochs, costs)
            plt.title("Training Cost")
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.show()
        return self.evaluate(X, Y)

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
