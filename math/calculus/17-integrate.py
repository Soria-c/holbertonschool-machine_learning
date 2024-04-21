#!/usr/bin/env python3
"""Calculus tasks"""


def poly_integral(poly, C=0):
    """Function that calculates the integral of a polynomial"""
    if not (isinstance(poly, list)) or len(poly) == 0:
        return None
    result = []
    for i, x in enumerate(poly):
        if (x != 0):
            value = x / (i + 1)
            result.append(int(value) if (value.is_integer()) else value)
    return [C, *result]
