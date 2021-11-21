"""Functions for creating synthetic task data"""

import numpy as np

from numpy.random import default_rng


class GlazeTask:

    def __init__(self, hazard_rate, noise, rng=default_rng()):

        # Set parameters
        self.hazard_rate = hazard_rate
        self.noise = noise
        self.rng = rng

        # Initialize side
        self.s_current = rng.choice([-1, 1])

    def _smp_twopoint(self, p):

        return self.rng.choice([1, -1], p=[p, 1 - p])

    def next_trial(self):

        # Side switches with probability equal to hazard rate
        self.s_current *= -self._smp_twopoint(self.hazard_rate)

        # Observation is side plus Gaussian noise
        x = self.s_current + self.rng.normal(0, self.noise)

        return (self.s_current, x)

    def sample_trials(self, n_trials):

        s_vals = np.full(n_trials, np.nan)
        x_vals = np.full(n_trials, np.nan)

        for i in range(n_trials):
            s_vals[i], x_vals[i] = self.next_trial()
        
        return s_vals, x_vals
 

class IBLTask:

    def __init__(self, hazard_rate, noise, alpha=0.8, rng=default_rng()):

        # Set parameters
        self.hazard_rate = hazard_rate
        self.noise = noise
        self.alpha = alpha
        self.rng = rng

        # Initialize block
        self.b_current = rng.choice([-1, 1])

    def _smp_twopoint(self, p):

        return self.rng.choice([1, -1], p=[p, 1-p])

    def next_trial(self):

        # Block switches with probability equal to hazard rate
        self.b_current *= -self._smp_twopoint(self.hazard_rate) 

        # Side probability depends on block
        s = self._smp_twopoint(self.alpha) * self.b_current

        # Observation is side plus Gaussian noise
        x = s + self.rng.normal(0, self.noise)

        return (self.b_current, s, x)

    def sample_trials(self, n_trials):

        b_vals = np.full(n_trials, np.nan)
        s_vals = np.full(n_trials, np.nan)
        x_vals = np.full(n_trials, np.nan)

        for i in range(n_trials):
            b_vals[i], s_vals[i], x_vals[i] = self.next_trial()
        
        return b_vals, s_vals, x_vals
