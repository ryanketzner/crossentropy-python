"""
gausslib.gaussian
-----------------
Multivariate Gaussian distribution: sampling, pdf, logpdf, and weighted MLE.
"""

import numpy as np
from scipy.stats import multivariate_normal
from numpy.linalg import LinAlgError


class Gaussian:
    """Multivariate Gaussian distribution.

    Attributes:
        mean (np.ndarray): 1D array of shape (d,) representing the mean.
        cov (np.ndarray): 2D array of shape (d, d) representing covariance.
    """

    def __init__(self, mean: np.ndarray, cov: np.ndarray):
        """
        Args:
            mean: 1D array of length d.
            cov: 2D, positive-semidefinite array of shape (d, d).
        """
        self.mean = np.asarray(mean, dtype=float)
        self.cov = np.asarray(cov, dtype=float)

    def sample(self, n: int) -> np.ndarray:
        """Draw n samples from this Gaussian.

        Args:
            n: Number of samples to draw.

        Returns:
            samples: Array of shape (n, d).
        """
        return np.random.multivariate_normal(self.mean, self.cov, size=n)

    def pdf(self, points: np.ndarray) -> np.ndarray:
        """Compute the probability density at given points.

        Args:
            points: Array of shape (n, d). Each row is a sample point.

        Returns:
            densities: 1D array of length n with pdf values.
        """
        points = np.atleast_2d(points)
        return multivariate_normal.pdf(points, mean=self.mean, cov=self.cov)

    def logpdf(self, points: np.ndarray) -> np.ndarray:
        """Compute the log-density at given points.

        Args:
            points: Array of shape (n, d). Each row is a sample point.

        Returns:
            log_densities: 1D array of length n with log-pdf values.

        Raises:
            ValueError: If the covariance matrix is singular.
        """
        points = np.atleast_2d(points)
        try:
            return multivariate_normal.logpdf(points, mean=self.mean, cov=self.cov)
        except LinAlgError:
            raise ValueError("Covariance matrix is singular.")

    @staticmethod
    def weighted_mle(samples: np.ndarray, weights: np.ndarray) -> "Gaussian":
        """Compute a Gaussian by weighted maximum-likelihood estimation.

        The weighted mean μ and covariance Σ are computed as:

            scaled_weights = (n_samples * weights) / sum(weights)
            μ = sum_i [ scaled_weights[i] * samples[i] ] / sum(scaled_weights)
            Σ = [ (samples - μ)ᵀ · ( (samples - μ) * scaled_weights[:, None] ) ] / sum(scaled_weights)

        Args:
            samples: Array of shape (n, d). Each row is a data point.
            weights: 1D array of length n (nonnegative).

        Returns:
            Gaussian: New distribution with computed mean and covariance.

        Raises:
            ValueError: If the sum of input weights is zero or dimensions mismatch.
        """
        samples = np.atleast_2d(samples)
        weights = np.asarray(weights, dtype=float).ravel()
        n, d = samples.shape
        if weights.shape[0] != n:
            raise ValueError("Length of weights must match number of samples.")

        total_w = weights.sum()
        if total_w == 0:
            raise ValueError("Sum of weights must be positive.")

        # Scale weights exactly as specified
        scaled = (n * weights) / total_w
        weight_sum = scaled.sum()

        # Compute weighted mean
        mean = (scaled[:, None] * samples).sum(axis=0) / weight_sum

        # Compute weighted covariance
        centered = samples - mean  # shape (n, d)
        cov = (centered.T @ (centered * scaled[:, None])) / weight_sum

        return Gaussian(mean, cov)