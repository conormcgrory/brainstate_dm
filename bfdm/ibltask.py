"""Functions for creating synthetic IBL task data"""

import numpy as np
from numpy.random import default_rng


def sample_twopoint(p: float, rng) -> int:
    """Sample 1 with probability p and -1 with probability 1-p."""

    return rng.choice([1, -1], p=[p, 1 - p])


def sample_biased_trials(n_trials: int, p_stay: float,
        p_side: float, noise: float,
        rng=default_rng()) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample synthetic trials from biased section of IBL task."""

    # Block, stimulus side and stimulus data
    b = np.full(n_trials, np.nan)
    s = np.full(n_trials, np.nan)
    x = np.full(n_trials, np.nan)

    # Initialize block randomly
    b[-1] = rng.choice([-1, 1])

    for i in range(n_trials):

        # Block stays the same with probability p_stay
        b[i] = b[i - 1] * sample_twopoint(p_stay, rng)

        # Side is same as block with probability p_side
        s[i] = b[i] * sample_twopoint(p_side, rng)

        # Observation is side plus Gaussian noise
        x[i] = s[i] + rng.normal(0, noise)

    return b, s, x


def sample_unbiased_trials(n_trials: int, noise: float,
        rng=default_rng()) -> tuple[np.ndarray, np.ndarray]:
    """Sample synthetic trials from unbiased section of IBL task."""

    # Stimulus side and stimulus data
    s = np.full(n_trials, np.nan)
    x = np.full(n_trials, np.nan)

    for i in range(n_trials):

        # Side is +1 or -1 with p=0.5
        s[i] = sample_twopoint(0.5, rng)

        # Observation is side plus Gaussian noise
        x[i] = s[i] + rng.normal(0, noise)

    return s, x
