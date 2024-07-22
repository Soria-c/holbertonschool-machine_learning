#!/usr/bin/env python3
"""Determinant using Laplace expansion"""


def minor_0(matrix, i, j):
    """
    Caculate the minor
    """
    return [row[:j] + row[j+1:] for row in (matrix[:i] + matrix[i+1:])]


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
        det += ((-1) ** c) * matrix[0][c] * determinant(minor_0(matrix, 0, c))
    return det


def minor(matrix):
    """
    Calculates the minor matrix of a matrix
    """
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    for row in matrix:
        if not isinstance(row, list):
            raise TypeError("matrix must be a list of lists")
        if len(row) != len(matrix):
            raise ValueError("matrix must be a non-empty square matrix")
    if len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a square matrix")

    if len(matrix) == 1:
        return [[1]]
    length = len(matrix)
    return [
        [determinant(minor_0(matrix, i, j)) for j in range(length)]
        for i in range(length)]
