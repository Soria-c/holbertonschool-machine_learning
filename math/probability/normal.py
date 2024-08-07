#!/usr/bin/env python3
"""Initialize Normal"""


class Normal:
    """Class to define Normal distribution"""
    def __init__(self, data=None, mean=0., stddev=1.):
        """Constructor"""
        self.e = 2.7182818285
        self.pi = 3.1415926536
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.stddev = float(stddev)
            self.mean = float(mean)
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.mean = float(sum(data)/len(data))
            self.variance = sum((x - self.mean) ** 2 for x in data) / len(data)
            self.stddev = float(self.variance ** 0.5)

    def erf(self, x):
        """
        Error function
        """
        return (2 / (self.pi ** 0.5)) *\
            (x - ((x ** 3) / 3) + ((x ** 5) / 10) -
             ((x ** 7) / 42) + ((x ** 9) / 216))

    def z_score(self, x):
        """
        Calculates the z-score of a given x-value.
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        Calculates the x-value of a given z-score.
        """
        return (z * self.stddev) + self.mean

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given x-value
        """
        return (1 / (self.stddev * (2 * self.pi) ** 0.5)) *\
            (self.e ** (-0.5 * ((x - self.mean) / self.stddev) ** 2))

    def cdf(self, x):
        """
        Calculates the value of the CDF for a given x-value
        """
        return 0.5 * (1 + self.erf((x - self.mean) / (self.stddev * 2 ** 0.5)))
