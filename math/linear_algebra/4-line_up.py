#!/usr/bin/env python3
"""This module defines a function named 'add_arrays'"""


def add_arrays(arr1, arr2):
    """Functions to calculate the sum of two arrays"""
    if len(arr1) == len(arr2):
        return [arr1[i] + arr2[i] for i in range(len(arr1))]
