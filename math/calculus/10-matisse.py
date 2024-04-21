#!/usr/bin/env python3
"""Calculus tasks"""


def poly_derivative(poly):
    """Function that calculates the derivative of a polynomial"""
    if not(isinstance(poly, list) and len(poly) == 0):
        return None
    result = []
    for i, x in enumerate(poly):
        if (i >= 1):
            result.append(i * x)
    result = [0] if len(result) == 0 else result
    return result
