#!/usr/bin/env python3
"""Slice Like A Ninja"""


def np_slice(matrix, axes={}):
    """
    Slices a matrix along specific axes
    """
    ndim = matrix.ndim
    indexing = tuple([slice(*axes.get(i, (None, None))) for i in range(ndim)])
    return matrix[indexing]
