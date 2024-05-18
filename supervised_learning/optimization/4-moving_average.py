#!/usr/bin/env python3
"""Mini-Batch"""


def moving_average(data, beta):
    """
    Calculates the weighted moving average of a data set
    ```
    Parameters:
    -----------
    data: list[,]
         list of data to calculate the moving average of
    beta: double
        weight used for the moving average
    Return:
    -------
    list containing the moving averages of data
    """
    v = 0

    def compute_exp_avg(d):
        """Computes exponential average"""
        nonlocal v
        v = (beta * v) + ((1 - beta) * d[1])
        bias_corrected = v / (1 - (beta ** (d[0] + 1)))
        return bias_corrected
    return list(map(compute_exp_avg, enumerate(data)))
