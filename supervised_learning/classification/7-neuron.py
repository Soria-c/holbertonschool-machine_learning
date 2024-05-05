#!/usr/bin/env python3
"""This module defines a representation of a Neuron"""
import numpy as np
import matplotlib.pyplot as plt


class Neuron:
    """
    A class to represent a Neuron

    ```
    Attributes
    ----------
    W : np.Array[Float]
        The weights vector for the neuron
    b : float
        The bias for the neuron.
    A : float
        The activated output of the neuron (prediction).
    """

    def __init__(self, nx: int):
        """
        Constructor

        ```
        Parameters
        ----------
        nx: int
            is the number of input features to the neuron
        """
        self.__validate(nx)
        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

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
        Calculates the forward propagation of the neuron

        ```
        Parameters
        ----------
        X: np.Array
            Input data
        Returns
        -------
        np.Array
            Activated output
        """
        self.__A = self.sig(np.matmul(self.__W, X) + self.__b)
        return self.__A

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
        return ((self.__A >= 0.50).astype(int), self.cost(Y, self.__A))

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neuron
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
        """
        dz = A - Y
        dw = np.matmul(dz, X.transpose()) / len(A[0])
        db = dz.mean()
        self.__W = self.__W - (alpha * dw)
        self.__b = self.__b - (alpha * db)

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
        verbose: bool
            defines whether or not to print information about the training
        graph: bool
            defines whether or not to graph information about the training
            once the training has completed.
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
        for i in range(iterations + 1):
            if verbose and (i % step) == 0:
                current_cost = self.cost(Y, self.__A)
                costs.append(current_cost)
                epochs.append(i)
                print(f"Cost after {i} iterations: {current_cost}")
            self.gradient_descent(X, Y, self.forward_prop(X), alpha)
        if graph is True:
            plt.plot(epochs, costs)
            plt.title("Training Cost")
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.show()
        return self.evaluate(X, Y)

    @property
    def W(self):
        """
        Getter for __W
        """
        return self.__W

    @property
    def b(self):
        """
        Getter for __b
        """
        return self.__b

    @property
    def A(self):
        """
        Getter for __A
        """
        return self.__A

    def __validate(self, nx: int):
        """
        Function to validate input
        ```
        Parameters
        ----------
        nx: int
            must be an integer
        """
        if (not isinstance(nx, int)):
            raise TypeError("nx must be an integer")
        if (nx < 1):
            raise ValueError("nx must be a positive integer")
