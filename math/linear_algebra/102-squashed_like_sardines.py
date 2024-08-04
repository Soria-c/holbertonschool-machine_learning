#!/usr/bin/env python3
"""Slice Like A Ninja"""


def matrix_shape(matrix):
    """Functions to calculate de dimensions of a matrix"""
    if (isinstance(matrix, list)):
        return [len(matrix), *matrix_shape(matrix[0])]
    return []


def __shape_comp(a_shape, b_shape, axis):
    """
    Check concatenation compatibility
    """
    if len(a_shape) != len(b_shape):
        return False
    for i in range(len(a_shape)):
        if i != axis and a_shape[i] != b_shape[i]:
            return False
    return True


def cat_matrices(mat1, mat2, axis=0):
    """
    Concatenates two matrices along a specific axis
    """
    shape1 = matrix_shape(mat1)
    shape2 = matrix_shape(mat2)
    if not __shape_comp(shape1, shape2, axis):
        return None
    return __concat(mat1, mat2, axis)


def __concat(mat1, mat2, axis):
    """Functions to concatenate two matrices in a given axis"""
    lenght = len(mat1)
    if axis == 0:
        return mat1 + mat2
    return [
        __concat(mat1[i], mat2[i], axis - 1)
        for i in range(lenght)
    ]
