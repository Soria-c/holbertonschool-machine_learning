#!/usr/bin/env python3
"""This module defines a function named 'matrix_transpose'"""


def matrix_transpose(matrix):
    """Functions to calculate the transpose of a matrix"""
    dimensions = [len(matrix), len(matrix[0])]
    transposed = []
    for i in range(dimensions[1]):
        inner = []
        for j in range(dimensions[0]):
            inner.append(matrix[j][i])
        transposed.append(inner)
    return transposed
