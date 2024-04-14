#!/usr/bin/env python3
"""This module defines a function named 'cat_matrices2D'"""


def cat_matrices2D(mat1, mat2, axis=0):
    """Functions to concatenate two matrices in a given axis"""
    if (axis == 0):
        return [*[list(i) for i in mat1], *[list(i) for i in mat2]]
    elif (axis == 1):
        return [[*mat1[i], *mat2[i]] for i in range(len(mat1))]
