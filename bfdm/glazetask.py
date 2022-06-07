"""Functions for creating synthetic task data"""

import numpy as np
from numpy.random import default_rng


def sample_twopoint(p: float, rng) -> int:
    """Sample 1 with probability p and -1 with probability 1-p."""

    return rng.choice([1, -1], p=[p, 1 - p])


def sample_trials(n_trials: int, p_stay: float, 
        noise: float, rng=default_rng()) -> tuple[np.ndarray, np.ndarray]:
    """Sample synthetic trials from Glaze task."""

    # Stimulus side and stimulus data
    s = np.full(n_trials, np.nan)
    x = np.full(n_trials, np.nan)

    # Initialize side randomly
    s[-1] = rng.choice([-1, 1])

    for i in range(n_trials):

        # Side stays the same with probability p_stay
        s[i] = s[i - 1] * sample_twopoint(p_stay, rng) 

        # Observation is side plus Gaussian noise
        x[i] = s[i] + rng.normal(0, noise)

    return s, x