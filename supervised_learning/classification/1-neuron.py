#!/usr/bin/env python3
"""This module defines a representation of a Neuron"""
import numpy as np


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
