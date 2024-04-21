#!/usr/bin/env python3
"""Calculus tasks"""


def summation_i_squared(n):
    """Function to calculate a sum"""
    if (isinstance(n, int)):
        if n == 1:
            return 1
        return n ** 2 + summation_i_squared(n - 1)
