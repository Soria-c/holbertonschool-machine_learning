#!/usr/bin/env python3
"""Calculus tasks"""


def poly_integral(poly, C=0):
    """Function that calculates the integral of a polynomial"""
    if not (isinstance(poly, list)) or (len(poly) == 0):
        return None
    if not (isinstance(C, [int, float])):
        return None
    if (len(poly) == 1) and (poly[0] == 0):
        return [C]
    result = []
    for i, x in enumerate(poly):
        value = x / (i + 1)
        result.append(int(value) if (value.is_integer()) else value)
    return [C, *result]
