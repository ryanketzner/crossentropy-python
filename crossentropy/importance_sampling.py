"""crossentropy.importance_sampling
----------------------------
Estimate P[s(X) < threshold] via importance sampling:

    - Draw samples from proposal q
    - Compute scores and identify hits
    - Compute importance weights w_i = p.pdf(x_i) / q.pdf(x_i)
    - Estimate mean, variance, relative error, and confidence interval
"""

import numpy as np


def compute_confidence_interval(mean, variance, n, z=1.96):
    """Compute the confidence interval for the estimated mean.

    Args:
        mean: Estimated mean.
        variance: Variance of the estimator.
        n: Number of samples.
        z: z-score for desired confidence level (default is 1.96 for 95%).

    Returns:
        A tuple (lower, upper) representing the confidence interval.
    """
    stderr = z * np.sqrt(variance / n)
    return [mean - stderr, mean + stderr]


def compute_relative_error(mean, variance, n, z=1.96):
    """Compute the relative error of the estimator.

    Args:
        mean: Estimated mean.
        variance: Variance of the estimator.
        n: Number of samples.
        z: z-score for desired confidence level (default is 1.96 for 95%).

    Returns:
        Relative error (standard error / mean).
    """
    stderr = z * np.sqrt(variance / n)
    return stderr / mean if mean != 0 else np.inf


def importance_sampling(
    p,
    q,
    score_fn,
    threshold: float,
    num: int = 10000,
    parallel: bool = False,
    z: float = 1.96,
):
    """Perform importance sampling to estimate P(s(X) < threshold).

    Args:
        p: Original distribution with method p.pdf(points) → (n,)
        q: Proposal distribution with methods:
            - q.sample(num) → array (num, d)
            - q.pdf(points) → array (n,)
        score_fn: Callable(x) → scalar score for 1D array x
        threshold: Scalar cutoff for “hit” event
        num: Number of samples (default=10000)
        parallel: If True, attempt parallel scoring (not implemented; falls back to serial)
        z: z-score used for confidence interval and relative error estimation

    Returns:
        mean_est: Estimated probability P(s(X) < threshold)
        var_est: Variance of the estimated probability
        RE: Relative error of the estimated probability
        interval: [lower, upper] confidence interval
        samples: Array of shape (num, d)
        scores: 1D array of length num
        hit_mask: Boolean array of length num indicating hits
    """
    samples = q.sample(num)
    n = samples.shape[0]

    scores = np.zeros(n, dtype=float)
    for i in range(n):
        scores[i] = score_fn(samples[i])

    hit_mask = scores < threshold
    hits = samples[hit_mask]

    if hits.shape[0] > 0:
        hit_weights = p.pdf(hits) / q.pdf(hits)
    else:
        hit_weights = np.zeros(0, dtype=float)

    weights_all = p.pdf(samples) / q.pdf(samples)
    indicator = hit_mask.astype(float)
    weighted_ind = weights_all * indicator

    mean_est = hit_weights.sum() / num
    var_est = ((weighted_ind - mean_est) ** 2).sum() / (num - 1)

    interval = compute_confidence_interval(mean_est, var_est, num, z=z)
    RE = compute_relative_error(mean_est, var_est, num, z=z)

    return mean_est, var_est, RE, interval, samples, scores, hit_mask