#!/usr/bin/env python3
"""Initialize Poisson"""


class Poisson:
    """Class to define to Poisson distribution"""
    def __init__(self, data=None, lambtha=1.):
        """Constructor"""
        self.e = 2.7182818285
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.lambtha = float(sum(data)/len(data))

    def factorial(self, n):
        """Function to compute factorial"""
        if n < 2:
            return 1
        else:
            return n * self.factorial(n-1)

    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of “successes”
        """
        k = int(k)
        return 0 if k < 0 else\
            ((self.e ** (-1 * self.lambtha)) * (self.lambtha ** k))\
            / self.factorial(k)

    def cdf(self, k):
        """
        Calculates the value of the CDF for a given number of “successes”
        """
        k = int(k)
        if k < 0:
            return 0
        return 0 if k < 0 else sum([self.pmf(i) for i in range(k + 1)])
