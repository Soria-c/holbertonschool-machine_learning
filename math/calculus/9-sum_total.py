#!/usr/bin/env python3
"""Calculus tasks"""
import numpy as np

def summation_i_squared(n):
    """Function to calculate a sum"""
    if (isinstance(n, int)):
        arr = np.arange(1, n+1)
        return np.power(arr, 2).sum()
