#!/usr/bin/env python3
"""This module defines a function named 'cat_matrices2D'"""


def cat_matrices2D(mat1, mat2, axis=0):
    """Functions to concatenate two matrices in a given axis"""
    l1 = len(mat1)
    l2 = len(mat2)
    if (l2 == 0 or l1 == 0):
        return
    if (axis == 0) and (len(mat1[0]) == len(mat2[0])):
        return [*[list(i) for i in mat1], *[list(i) for i in mat2]]
    if (axis == 1) and (l1 == l2):
        return [[*mat1[i], *mat2[i]] for i in range(l1)]
