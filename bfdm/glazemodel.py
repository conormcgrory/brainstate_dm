"""Bayesian filtering model from Glaze et al., 2015"""

import numpy as np
import scipy.optimize as opt

from numpy.random import default_rng
from scipy.special import logit, expit


def phi(a, b):
    """Function used to recursively compute prior term"""

    return np.logaddexp(0, a + b) - np.logaddexp(a, b)

def log_prior(r_prev, logit_h):
    """Log-prior ratio for single time point"""

    return phi(r_prev, -logit_h)

def log_lik(x, w_0, w_1):
    """Log-likelihood ratio for single time point"""

    return w_0 + w_1 * x

def log_pos_full(x, logit_h, w_0, w_1):
    """Recursively compute log-posterior ratio for all time points"""
    
    r = np.full(x.shape[0], np.nan)
    r[-1] = 0

    for t in range(x.shape[0]):
        r[t] = log_lik(x[t], w_0, w_1) + log_prior(r[t - 1], logit_h)
        
    return r


class GlazeAgent:
    """Agent implementing Bayesian filtering for Glaze task."""

    def __init__(self, h, w_0, w_1, rng=default_rng()):
        
        # Hazard rate
        self.h = h

        # Logit of hazard rate
        self.logit_h = logit(h)

        # Bias term
        self.w_0 = w_0
        
        # Weight for contrast
        self.w_1 = w_1

        # Random number generator (for sampling)
        self.rng = rng

    def decision_function(self, x):
        """Compute log-posterior ratio for this agent on given input"""

        return log_pos_full(x, self.logit_h, self.w_0, self.w_1)
 
    def sample(self, x, return_r=False):
        """Sample choices from agent for given input."""

        # Log posterior ratio for inputs
        r = self.decision_function(x)

        # Probability of choosing 1 (right)
        p = expit(r)

        # Generate samples
        y = 2 * self.rng.binomial(1, p) - 1
        
        if return_r:
            return y, r
        else:
            return y
 

class GlazeModel:
    """Behavior model using Bayesian filtering on Glaze task."""
    
    def __init__(self):

        # Hazard rate
        self.h = None
        
        # Logit of hazard rate
        self.logit_h = None
        
        # Bias term
        self.w_0 = None
        
        # Weight for contrast
        self.w_1 = None
        
        # Result from optimization
        self.opt_result = None
            
    @staticmethod
    def neg_LL(theta, x, y):
        """Negative log-likelihood function minimized to fit model."""
    
        r = log_pos_full(x, theta[0], theta[1], theta[2])
        y_bin = (y + 1) / 2
    
        return np.sum(np.logaddexp(0, r) - y_bin * r)

    def fit(self, x, y):
        """Fit model to inputs and choices."""
        
        # Initial parameter valuesj
        theta_0 = np.array([0, 0, 0])

        # Estimate parameters by minimizing log-likelihood
        res = opt.minimize(
            GlazeModel.neg_LL, 
            theta_0, 
            args=(x, y), 
            method='BFGS', 
            options={'gtol':1e-2}
        )
        
        # Set parameter values
        self.logit_h = res.x[0]
        self.h = expit(self.logit_h)
        self.w_0 = res.x[1]
        self.w_1 = res.x[2]

        # Store optimization result
        self.opt_result = res

    def decision_function(self, x):
        """Compute decision function (log-posterior ratio) for fit model."""

        return log_pos_full(x, self.logit_h, self.w_0, self.w_1)

    def predict(self, x):
        """Predict choice for given inputs."""

        return np.sign(self.decision_function(x))

    def predict_proba(self, x):
        """Return choice probabilities (y=1) for given inputs."""

        return expit(self.decision_function(x))
