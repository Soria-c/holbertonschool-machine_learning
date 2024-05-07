#!/usr/bin/env python3
"""This module defines a representation of a Deep Neural Network"""
import numpy as np


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
    def __init__(self, nx, layers):
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

    @property
    def L(self):
        """
        Getter for __L
        """
        return self.__L

    @property
    def L(self):
        """
        Getter for __cache
        """
        return self.__cache

    @property
    def L(self):
        """
        Getter for __weights
        """
        return self.__weights
