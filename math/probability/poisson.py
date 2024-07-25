#!/usr/bin/env python3
"""Initialize Poisson"""


class Poisson:
    """Class to define to Poisson distribution"""
    def __init__(self, data=None, lambtha=1.):
        """Constructor"""
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not(isinstance(data, list)) or len(data) < 2:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(sum(data)/len(data))
