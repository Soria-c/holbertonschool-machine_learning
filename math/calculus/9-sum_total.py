#!/usr/bin/env python3
"""Calculus tasks"""


def summation_i_squared(n):
    """Function to calculate a sum"""
    if (isinstance(n, int)) and n > 0:
        return n * (n + 1) * (2 * n + 1) / 6
