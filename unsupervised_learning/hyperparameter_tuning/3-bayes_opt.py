#!/usr/bin/env python3
"""Initialize Bayesian Optimization"""
import numpy as np

GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
    Performs Bayesian optimization on a noiseless 1D Gaussian process
    """
    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        """
        Initializes Bayesian Optimization on a Gaussian Process

        Parameters:
        - f: function, the black-box function to be optimized
        - X_init: numpy.ndarray of shape (t, 1), inputs already sampled
            with the black-box function
        - Y_init: numpy.ndarray of shape (t, 1), outputs of the black-box
            function for each input in X_init
        - bounds: tuple of (min, max), the bounds of the space in which to
            look for the optimal point
        - ac_samples: int, the number of samples that should be analyzed
            during acquisition
        - l: float, length parameter for the kernel (default: 1)
        - sigma_f: float, standard deviation of the output (default: 1)
        - xsi: float, exploration-exploitation factor
            for acquisition (default: 0.01)
        - minimize: bool, determines whether optimization should
            be performed for minimization (default: True)

        Attributes:
        - f: the black-box function
        - gp: instance of the GaussianProcess class
        - X_s: numpy.ndarray of shape (ac_samples, 1),
            acquisition sample points
        - xsi: exploration-exploitation factor
        - minimize: bool for minimization versus maximization
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.bounds = bounds
        self.xsi = xsi
        self.minimize = minimize

        # Generate acquisition sample points (X_s)
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)
