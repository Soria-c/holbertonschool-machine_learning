#!/usr/bin/env python3
"""This module defines a function named 'np_cat'"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """Function to concatenate two matrices in a given axis"""
    return np.concatenate((mat1, mat2), axis=axis)
