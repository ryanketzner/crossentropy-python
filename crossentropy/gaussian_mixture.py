"""
crossentropy.gaussian_mixture
-------------------------
Gaussian Mixture Model (GMM) with EM fitting (unweighted and weighted).
"""

import numpy as np
from sklearn.cluster import KMeans
from .gaussian import Gaussian


class GaussianMixture:
    """Gaussian Mixture Model (GMM).

    Attributes:
        components (list[Gaussian]): List of Gaussian components, length K.
        weights (np.ndarray): Mixing proportions, 1D array of shape (K,).
    """

    def __init__(self, components: list[Gaussian], weights: np.ndarray):
        """
        Args:
            components: List of Gaussian objects (length K).
            weights: 1D array of length K, must sum to 1.
        """
        self.components = components
        self.weights = np.asarray(weights, dtype=float).ravel()
        if len(self.components) != self.weights.shape[0]:
            raise ValueError("Number of components and weights must match.")
        if not np.isclose(self.weights.sum(), 1.0):
            raise ValueError("Mixing weights must sum to 1.")

    def sample(self, n: int) -> np.ndarray:
        """Draw n samples from the mixture.

        Args:
            n: Number of samples.

        Returns:
            samples: Array of shape (n, d).
        """
        K = len(self.components)
        # Randomly select component index for each sample
        comp_idxs = np.random.choice(K, size=n, p=self.weights)
        d = self.components[0].mean.shape[0]
        samples = np.zeros((n, d))

        for k in range(K):
            idx_k = np.where(comp_idxs == k)[0]
            if idx_k.size > 0:
                samples[idx_k] = self.components[k].sample(idx_k.size)
        return samples

    def pdf(self, points: np.ndarray) -> np.ndarray:
        """Compute the mixture pdf at each point.

        Args:
            points: Array of shape (n, d).

        Returns:
            densities: 1D array of length n.
        """
        points = np.atleast_2d(points)
        n = points.shape[0]
        K = len(self.components)
        densities = np.zeros((n, K))

        for k, comp in enumerate(self.components):
            densities[:, k] = self.weights[k] * comp.pdf(points)
        return densities.sum(axis=1)

    @staticmethod
    def mle(samples: np.ndarray, num_components: int = 1) -> "GaussianMixture":
        """Fit a GMM by unweighted EM with k-means initialization.

        Procedure:
          1. Run k-means with K = num_components on 'samples'.
          2. Initialize each component's mean & cov by weighted_mle on its cluster.
          3. Iterate EM exactly 1000 times:
               - E-step: compute responsibilities r[i,k]
               - M-step: update each Gaussian via weighted_mle using r[:,k] as weights,
                 and update mixing coefficients as sum(r[:,k]) / n.

        Args:
            samples: Array of shape (n, d).
            num_components: Number of mixture components, K (default=1).

        Returns:
            GaussianMixture: Fitted model.
        """
        samples = np.atleast_2d(samples)
        n, d = samples.shape

        # All-sample weights (unweighted EM)
        weights_vec = np.ones(n, dtype=float)

        # K-means initialization
        kmeans = KMeans(n_clusters=num_components, init="random", n_init=1)
        labels = kmeans.fit_predict(samples)  # array of shape (n,)

        # Build initial components and uniform mixing weights
        components = []
        mixing = np.full(num_components, 1.0 / num_components, dtype=float)
        for k in range(num_components):
            cluster_samples = samples[labels == k]
            cluster_weights = np.ones(cluster_samples.shape[0], dtype=float)
            comp = Gaussian.weighted_mle(cluster_samples, cluster_weights)
            components.append(comp)

        # EM loop for exactly 1000 iterations
        for _ in range(1000):
            resp = GaussianMixture._compute_responsibilities(components, samples, mixing)
            components, mixing = GaussianMixture._m_step(samples, weights_vec, resp)

        return GaussianMixture(components, mixing)

    @staticmethod
    def weighted_mle(
        samples: np.ndarray, 
        weights: np.ndarray, 
        num_components: int = 1
    ) -> "GaussianMixture":
        """Fit a GMM by weighted EM with k-means initialization.

        Procedure:
          1. Run k-means with K = num_components on 'samples'.
          2. Initialize each component's mean & cov by weighted_mle on its cluster.
          3. Iterate EM exactly 1000 times:
               - E-step: compute responsibilities r[i,k]
               - M-step: update each Gaussian via weighted_mle using w[i] * r[i,k] as weights,
                 and update mixing coefficients as sum(w[i]*r[i,k]) / sum(w).

        Args:
            samples: Array of shape (n, d).
            weights: 1D array of length n (nonnegative).
            num_components: Number of mixture components, K (default=1).

        Returns:
            GaussianMixture: Fitted model.
        """
        samples = np.atleast_2d(samples)
        weights = np.asarray(weights, dtype=float).ravel()
        n, d = samples.shape
        if weights.shape[0] != n:
            raise ValueError("Length of weights must match number of samples.")

        # K-means initialization (ignores weights during clustering)
        kmeans = KMeans(n_clusters=num_components, init="random", n_init=1)
        labels = kmeans.fit_predict(samples)

        # Build initial components and uniform mixing weights
        components = []
        mixing = np.full(num_components, 1.0 / num_components, dtype=float)
        for k in range(num_components):
            cluster_samples = samples[labels == k]
            cluster_weights = weights[labels == k]
            comp = Gaussian.weighted_mle(cluster_samples, cluster_weights)
            components.append(comp)

        # EM loop for exactly 1000 iterations
        for _ in range(1000):
            resp = GaussianMixture._compute_responsibilities(components, samples, mixing)
            components, mixing = GaussianMixture._m_step(samples, weights, resp)

        return GaussianMixture(components, mixing)

    @staticmethod
    def _compute_responsibilities(
        components: list[Gaussian], 
        samples: np.ndarray, 
        mixing: np.ndarray
    ) -> np.ndarray:
        """Compute posterior responsibilities r[i,k] ∝ mixing[k] * p_k(x_i).

        Args:
            components: List of Gaussian components (length K).
            samples: Array of shape (n, d).
            mixing: 1D array of mixing weights (length K).

        Returns:
            resp: Array of shape (n, K) where each row sums to 1.
        """
        n, _ = samples.shape
        K = len(components)
        log_probs = np.zeros((n, K))

        for k in range(K):
            # log(π_k) + log p_k(x_i)
            try:
                log_probs[:, k] = np.log(mixing[k]) + components[k].logpdf(samples)
            except ValueError:
                log_probs[:, k] = -np.inf

        # Numerically stable normalization via log-sum-exp
        max_log = np.max(log_probs, axis=1, keepdims=True)
        shifted = np.exp(log_probs - max_log)
        sum_shifted = shifted.sum(axis=1, keepdims=True)
        log_sum_exp = max_log + np.log(sum_shifted)

        log_resp = log_probs - log_sum_exp
        return np.exp(log_resp)

    @staticmethod
    def _m_step(
        samples: np.ndarray, 
        weights: np.ndarray, 
        responsibilities: np.ndarray
    ) -> tuple[list[Gaussian], np.ndarray]:
        """M-step: update each Gaussian and mixing weights.

        For each component k:
          w_total[k] = sum_i [ weights[i] * responsibilities[i,k] ]
          new_mixing[k] = w_total[k] / sum(weights)
          new_comp[k] = Gaussian.weighted_mle(samples, weights * responsibilities[:,k])

        Args:
            samples: Array of shape (n, d).
            weights: 1D array of length n. (Sample weights; unweighted EM uses all ones.)
            responsibilities: 2D array shape (n, K).

        Returns:
            (components, new_mixing):
                components: List of updated Gaussians (length K).
                new_mixing: 1D array of length K (sums to 1).
        """
        n, _ = samples.shape
        K = responsibilities.shape[1]
        total_weights = weights[:, None] * responsibilities  # shape (n, K)
        weight_sum = weights.sum()

        new_components = []
        new_mixing = np.zeros(K, dtype=float)

        for k in range(K):
            w_k = total_weights[:, k]
            comp = Gaussian.weighted_mle(samples, w_k)
            new_components.append(comp)
            new_mixing[k] = w_k.sum() / weight_sum

        return new_components, new_mixing