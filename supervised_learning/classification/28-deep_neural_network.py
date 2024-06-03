#!/usr/bin/env python3
"""This module defines a representation of a Deep Neural Network"""
import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    """
    Defines a deep neural network performing binary classification
    ```
    Attributes
    ----------
    L: int
        The number of layers in the neural network.
    cache: dict
        A dictionary to hold all intermediary values of the network.
    weights: dict
        A dictionary to hold all weights and biased of the network.
    """
    def __init__(self, nx, layers, activation='sig'):
        """
        Constructor

        ```
        Parameters
        ----------
        nx: int
            is the number of input features to the neuron
        layers: list
            is a list representing the number of nodes in each
            layer of the network
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if (nx < 1):
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) == 0 or min(layers) < 1\
                or set(map(type, layers)) != {int}:
            raise TypeError("layers must be a list of positive integers")
        if activation not in ["sig", "tanh"]:
            raise ValueError("activation must be 'sig' or 'tanh'")
        self.__activation = activation
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for hidden_layer_number, current_neuron_amount in enumerate(layers):
            if hidden_layer_number == 0:
                self.__init_weights(nx, current_neuron_amount,
                                    hidden_layer_number)
            else:
                self.__init_weights(layers[hidden_layer_number - 1],
                                    current_neuron_amount, hidden_layer_number)

    def __init_weights(self, prev_neuron_amount, current_neuron_amount,
                       hidden_layer_number):
        """
        Function to initialize the weights
        Parameters
        ----------
        prev_node_amount: int
            number of neurons in the previous layer
        current_neuron_amount: int
            number of neurons in the current layer
        hidden_layer_number: int
            index of the current layer
        """
        weight_matrix = np.random.randn(current_neuron_amount,
                                        prev_neuron_amount)
        he_et_el = np.sqrt(2 / prev_neuron_amount)
        self.__weights[f"W{hidden_layer_number + 1}"] = weight_matrix\
            * he_et_el
        self.__weights[f"b{hidden_layer_number + 1}"] = np\
            .zeros(shape=(current_neuron_amount, 1))

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

    def tanh(self, x):
        """Tanh activation"""
        return np.tanh(x)

    def dtanh(self, x):
        """Derivative of Tanh activation"""
        return 1 - (x ** 2)

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        expZ = np.exp(x - np.max(x, axis=0, keepdims=True))
        return expZ / expZ.sum(axis=0, keepdims=True)

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
        Calculates the forward propagation of the deep neural network
        ```
        Parameters
        ----------
        X : np.Array
            Contains the input data
        """
        self.__cache["A0"] = X
        for i in range(self.__L):
            r = np.matmul(
                self.__weights[f"W{i +1}"], self.__cache[f"A{i}"])\
                + self.__weights[f"b{i +1}"]
            if (i + 1 == self.__L):
                result = self.softmax(r)
            else:
                result = self.sig(r) if self.__activation == "sig"\
                    else self.tanh(r)
            self.__cache[f"A{i+1}"] = result
        return result, self.__cache

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
        return -np.mean(np.sum(Y * np.log(A), axis=0))

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
        result = self.forward_prop(X)[0]
        argmax = np.argmax(result, axis=0)
        p = np.identity(result.shape[0])[argmax].transpose()
        return (p, self.cost(Y, result))

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neuron
        ```
        Parameters
        ----------
        Y: np.Array (1, m)
            contains the correct labels for the input data
        cache: dict
            is a dictionary containing all the intermediary values of the
            network
        alpha: float
            learning rate
        """
        prev_dz = None
        prev_w = None
        m = len(Y[0])
        for i in range(self.__L, 0, -1):
            if i == self.__L:
                dz = cache[f"A{i}"] - Y
            else:
                if self.__activation == "sig":
                    dz = np.matmul(prev_w.transpose(), prev_dz) *\
                        self.dzig(cache[f"A{i}"])
                else:
                    dz = np.matmul(prev_w.transpose(), prev_dz) *\
                        self.dtanh(cache[f"A{i}"])
            dw = np.matmul(dz, cache[f"A{i - 1}"].transpose()) / m
            db = dz.mean(axis=1, keepdims=True)
            prev_dz = dz
            prev_w = self.__weights[f"W{i}"]
            self.__weights[f"W{i}"] -= (alpha * dw)
            self.__weights[f"b{i}"] -= (alpha * db)

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
            a, _ = self.forward_prop(X)
            if verbose and (i % step) == 0:
                current_cost = self.cost(Y, a)
                costs.append(current_cost)
                epochs.append(i)
                print(f"Cost after {i} iterations: {current_cost}")
            self.gradient_descent(Y, self.__cache, alpha)

        if graph is True:
            plt.plot(epochs, costs)
            plt.title("Training Cost")
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.show()
        return self.evaluate(X, Y)

    def save(self, filename):
        """
        Saves the instance object to a file in pickle format
        Parameters:
        -----------
        filename: str
            file to which the object should be saved
        """

        with open(filename if len(fn := filename.split(".")) > 1 and
                  fn[-1] == "pkl" else filename+".pkl", "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """
        Loads a pickled DeepNeuralNetwork object
        Parameters:
        -----------
        filename: str
            file from which the object should be loaded
        """
        try:

            with open(filename, "rb") as f:
                return pickle.load(f)
        except Exception:
            return

    @property
    def L(self):
        """
        Getter for __L
        """
        return self.__L

    @property
    def cache(self):
        """
        Getter for __cache
        """
        return self.__cache

    @property
    def weights(self):
        """
        Getter for __weights
        """
        return self.__weights

    @property
    def activation(self):
        """
        Getter for __activation
        """
        return self.__activation
