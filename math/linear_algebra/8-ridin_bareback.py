#!/usr/bin/env python3
"""This module defines functions to calculate matrix multiplication'"""


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


def mat_mul(mat1, mat2):
    """Function to calculate matrix multiplication"""
    result = []
    if (len(mat1[0]) == len(mat2)):
        for i in mat1:
            inner = []
            for j in list(matrix_transpose(mat2)):
                inner.append(sum([x * y for x, y in zip(i, j)]))
            result.append(inner)
    return result
