#!/usr/bin/env python3
"""Initialize Normal"""


class Normal:
    """Class to define Normal distribution"""
    def __init__(self, data=None, mean=0., stddev=1.):
        """Constructor"""
        self.e = 2.7182818285
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
