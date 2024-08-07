#!/usr/bin/env python3
"""Binomial distribution"""


class Binomial:
    """Class to define the Binomial distribution"""
    def __init__(self, data=None, n=1, p=0.5):
        """Constructor"""
        if data is None:
            if n <= 0:
                raise ValueError('n must be a positive value')
            if p <= 0 or p >= 1:
                raise ValueError('p must be greater than 0 and less than 1')
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)
            self.p = 1 - variance / mean
            self.n = round(mean / self.p)
            self.p = mean / self.n

    def factorial(self, n):
        """Function to compute factorial"""
        if n < 2:
            return 1
        else:
            return n * self.factorial(n-1)

    def nCk(self, k):
        """
        Compute combinatory
        """
        return (self.factorial(self.n) /
                (self.factorial(k) * self.factorial(self.n - k)))

    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of “successes”
        """
        k = int(k)
        if k < 0 or k > self.n:
            return 0
        return self.nCk(k) * (self.p ** k) * ((1 - self.p) ** (self.n - k))

    def cdf(self, k):
        """
        Calculates the value of the CDF for a given number of “successes”
        """
        k = int(k)
        if k < 0 or k > self.n:
            return 0
        return sum(self.pmf(i) for i in range(k + 1))
