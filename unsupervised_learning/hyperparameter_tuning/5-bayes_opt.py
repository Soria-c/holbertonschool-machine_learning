#!/usr/bin/env python3
"""Initialize Bayesian Optimization"""


import numpy as np
GP = __import__('2-gp').GaussianProcess
from scipy.stats import norm


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

    def acquisition(self):
        """
        Calculates the next best sample location using the
        Expected Improvement (EI) acquisition function.

        Returns:
        - X_next: numpy.ndarray of shape (1,), the next best point to sample
        - EI: numpy.ndarray of shape (ac_samples,),
            the expected improvement for each potential sample
        """
        mu, sigma = self.gp.predict(self.X_s)

        if self.minimize:
            mu_sample_opt = np.min(self.gp.Y)
            imp = mu_sample_opt - mu - self.xsi
        else:
            mu_sample_opt = np.max(self.gp.Y)
            imp = mu - mu_sample_opt - self.xsi

        # with np.errstate(divide='warn'):
        Z = imp / sigma
        EI = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        EI[sigma == 0.0] = 0.0  # Handle the case where sigma is zero

        X_next = self.X_s[np.argmax(EI)].reshape(1, 1)
        return X_next, EI

    def optimize(self, iterations=100):
        """
        Optimizes the black-box function over a number of iterations.

        Parameters:
        - iterations: int, maximum number of iterations to perform

        Returns:
        - X_opt: numpy.ndarray of shape (1,), the optimal point found
        - Y_opt: numpy.ndarray of shape (1,), the optimal function value found
        """
        # Iterate up to the specified number of iterations
        for _ in range(iterations):
            # Get the next sample point using the acquisition function
            X_next, _ = self.acquisition()

            # If the next proposed point has already been sampled, stop early
            if np.any(np.isclose(self.gp.X, X_next)):
                break

            # Evaluate the black-box function at the new sample point
            Y_next = self.f(X_next)

            # Update the Gaussian process model with the new sample
            self.gp.update(X_next, Y_next)

        # After optimization, find the optimal point and value
        if self.minimize:
            # If minimizing, the optimal point is the one with
            # the lowest function value
            idx_opt = np.argmin(self.gp.Y)
        else:
            # If maximizing, the optimal point is the one with
            # the highest function value
            idx_opt = np.argmax(self.gp.Y)

        # Extract the optimal point and its corresponding function value
        X_opt = self.gp.X[idx_opt]
        Y_opt = self.gp.Y[idx_opt]

        return X_opt, Y_opt
