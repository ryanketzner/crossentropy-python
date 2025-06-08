#!/usr/bin/env python3
"""
example_scaledgaussian.py

Perform Cross‐Entropy + Importance Sampling with a single ScaledGaussian proposal.
Find the probability that a ScaledGaussian random variable lies in a small circle.
Uses the scaled gaussian rather than the gaussian.

See section III. D. of "Monte Carlo Orbital Conjunction Analysis using the
Cross-Entropy Method with ScaledGaussian Mixture"

True probability is approximately 1.029e-05. This true probability should (~95% of the time,
under somewhat shaky assumptions about normality) fall within the confidence interval output
by this example.
"""

import numpy as np
import matplotlib.pyplot as plt

from crossentropy.scaledgaussian import ScaledGaussian
from crossentropy.cross_entropy import cross_entropy
from crossentropy.importance_sampling import importance_sampling


def main():
    # Define target ScaledGaussian distribution p
    mean = np.array([0.0, 0.0])
    cov = np.array([[2.0, 0.0],
                    [0.0, 1.0]])
    p = ScaledGaussian(mean, cov)

    # Scoring function: Euclidean distance from center
    center = np.array([5.0, 2.5])
    threshold = 0.5

    def score_fn(x: np.ndarray) -> float:
        return np.linalg.norm(x - center)

    # Cross‐Entropy parameters
    num_samples = 100000
    quantile = 0.2
    max_iters = 10
    parallel = False
    extras = 0

    # Update function for CE: weighted MLE of ScaledGaussian
    # Note, unlike for the Gaussian class, the weighted_mle function
    # for ScaledGaussian is NOT static. Make sure to use the weighted_mle
    # function corresponding to the target distribution for the problem at hand p 
    def update_fn(samples: np.ndarray, weights: np.ndarray) -> ScaledGaussian:
        return p.weighted_mle(samples, weights)

    # Run Cross‐Entropy to find proposal q
    q_dist, n_iters, completed = cross_entropy(
        p=p,
        update_fn=update_fn,
        score_fn=score_fn,
        threshold=threshold,
        num=num_samples,
        quantile=quantile,
        max_iters=max_iters,
        parallel=parallel,
        extras=extras
    )

    print(f"Cross‐Entropy completed in {n_iters} iterations. Success = {completed}")

    # Run Importance Sampling with final proposal q_dist
    mean_est, var_est, RE, interval, samples, scores, hit_mask = importance_sampling(
        p=p,
        q=q_dist,
        score_fn=score_fn,
        threshold=threshold,
        num=num_samples,
        parallel=parallel
    )

    print(f"Estimated P(score < {threshold}): {mean_est}")
    print(f"Estimator variance: {var_est}")
    print(f"Relative error: {RE}")
    print(f"95% CI: {interval}")

    # Scatter plot: all samples vs. hits
    hit_samples = samples[hit_mask]
    plt.figure(figsize=(6, 5))
    plt.scatter(samples[:, 0], samples[:, 1], s=5, alpha=0.3, label="All samples")
    plt.scatter(hit_samples[:, 0], hit_samples[:, 1],
                c="red", s=5, label="Hits (score < threshold)")
    plt.xlabel("x₁")
    plt.ylabel("x₂")
    plt.title("Importance Sampling with Single ScaledGaussian Proposal")
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()