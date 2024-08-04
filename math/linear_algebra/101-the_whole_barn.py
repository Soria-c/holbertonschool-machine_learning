#!/usr/bin/env python3
"""Slice Like A Ninja"""


def matrix_shape(matrix):
    """Functions to calculate de dimensions of a matrix"""
    if (isinstance(matrix, list)):
        return [len(matrix), *matrix_shape(matrix[0])]
    return []


def __sum(mat1, mat2, dim):
    """"
    Recursion to sum element wise
    """
    ndim = len(dim)
    return [
        mat1[d] + mat2[d] if ndim == 1
        else __sum(mat1[d], mat2[d], dim[1:])
        for d in range(dim[0])
    ]


def add_matrices(mat1, mat2):
    """
    Adds two matrices
    """
    dim = matrix_shape(mat1)
    if (dim != matrix_shape(mat2)):
        return None
    return __sum(mat1, mat2, dim)
