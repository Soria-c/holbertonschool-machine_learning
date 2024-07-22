#!/usr/bin/env python3
"""Determinant using Laplace expansion"""


def minor(matrix, j):
    """
    Caculate the minor
    """
    return [row[:j] + row[j+1:] for row in (matrix[:0] + matrix[1:])]


def determinant(matrix):
    """
    Calculates the determinant of a matrix
    """

    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    for row in matrix:
        if not isinstance(row, list):
            raise TypeError("matrix must be a list of lists")
    if len(matrix[0]) == 0:
        return 1
    if len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a square matrix")
    if len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for c in range(len(matrix)):
        det += ((-1) ** c) * matrix[0][c] * determinant(minor(matrix, c))
    return det
