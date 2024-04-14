#!/usr/bin/env python3
"""This module defines a function named 'matrix_shape'"""


def matrix_shape(matrix):
    """Functions to calculate de dimensions of a matrix"""
    if (isinstance(matrix, list)):
        return [len(matrix), *matrix_shape(matrix[0])]
    return []
