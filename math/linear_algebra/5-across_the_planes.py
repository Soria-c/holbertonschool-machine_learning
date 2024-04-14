#!/usr/bin/env python3
"""This module defines the functions named 'add_arrays' and 'add_matrices2D'"""


def add_matrices2D(mat1, mat2):
    """Functions to calculate the sum of two matrices"""
    if len(mat1) == len(mat2) and len(mat1[0]) == len(mat2[0]):
        return [add_arrays(mat1[i], mat2[i]) for i in range(len(mat1))]


def add_arrays(arr1, arr2):
    """Functions to calculate the sum of two arrays"""
    return [arr1[i] + arr2[i] for i in range(len(arr1))]
