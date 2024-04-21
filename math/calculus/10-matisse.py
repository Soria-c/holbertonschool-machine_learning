#!/usr/bin/env python3
"""Calculus tasks"""


def poly_derivative(poly):
    """Function that calculates the derivative of a polynomial"""
    if not(isinstance(poly, list) or len(poly) == 0):
        return None
    result = []
    degree = False
    for i, x in enumerate(poly):
        value = i * x
        if (value != 0):
            degree = True
            result.append(value)
        elif (degree):
            result.append(value)
    result = [0] if len(result) == 0 else result
    return result
