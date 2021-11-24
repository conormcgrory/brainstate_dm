"""Bayesian filtering for IBL task"""

import numpy as np
import scipy.optimize as opt

from numpy.random import default_rng
from scipy.special import logit, expit


def phi(a, b):
    """Function used to recursively compute prior term"""

    return np.logaddexp(0, a + b) - np.logaddexp(a, b)
 
def block_log_prior(q_prev, logit_h):
    """Log-prior ratio for block"""

    return phi(q_prev, -logit_h)

def block_log_lik(side, logit_a):
    """Log-likelihood ratio for block given side value."""

    return side * logit_a

def side_log_prior(q_prev, logit_h, logit_a):
    """Log-prior ratio for side given posterior over previous block"""

    return phi(q_prev, phi(-logit_h, logit_a))

def side_log_lik(x, w_0, w_1):
    """Log-likelihood ratio for side given single input"""
    
    return w_0 + w_1 * x

def log_pos_full(x, s, logit_h, logit_a, w_0, w_1):
    """Recursively compute log-posterior ratios for all time points"""
    
    q = np.full_like(x, np.nan)
    r = np.full_like(x, np.nan)
        
    q[-1] = 0
    r[-1] = 0

    for t in range(x.shape[0]):
            
        # Use previous block prior and input to compute log-posterior over s_t
        r[t] = side_log_lik(x[t], w_0, w_1) + side_log_prior(q[t - 1], logit_h, logit_a)
            
        # After observing s[t] in feedback, update log-posterior over block
        q[t] = block_log_lik(s[t], logit_a) + block_log_prior(q[t - 1], logit_h)
        

    return r, q


class IBLAgent:
    """Agent implementing Bayesian filtering for IBL task."""

    def __init__(self, h, a, w_0, w_1, rng=default_rng()):
        
        # Hazard rate
        self.h = h
        self.logit_h = logit(h)
        
        # Side probability
        self.a = a
        self.logit_a = logit(a)

        # Bias term
        self.w_0 = w_0
        
        # Weight for contrast
        self.w_1 = w_1

        # Random number generator (for sampling)
        self.rng = rng

    def decision_function(self, x, s):
        """Compute log-posterior ratio for this agent on given input"""

        r, _ = log_pos_full(x, s, self.logit_h, self.logit_a, self.w_0, self.w_1)
        
        return r

    def sample(self, x, s, return_rq=False):
        """Sample choices from agent for given input."""

        # Log posterior ratio for inputs
        r, q = log_pos_full(x, s, self.logit_h, self.logit_a, self.w_0, self.w_1)

        # Probability of choosing 1 (right)
        p = expit(r)

        # Generate samples
        y = 2 * self.rng.binomial(1, p) - 1
        
        if return_rq:
            return y, r, q
        else:
            return y


class IBLModel:
    """Behavior model using Bayesian filtering on IBL task."""
    
    def __init__(self):

        # Hazard rate
        self.h = None
        self.logit_h = None

        # Side probability
        self.a = None
        self.logit_a = None
        
        # Bias term
        self.w_0 = None
        
        # Weight for contrast
        self.w_1 = None
        
        # Result from optimization
        self.opt_result = None
            
    @staticmethod
    def neg_LL(theta, x, s, y):
        """Negative log-likelihood function minimized to fit model."""
    
        r, _ = log_pos_full(x, s, theta[0], theta[1], theta[2], theta[3])

        y_bin = (y + 1) / 2
    
        return np.sum(np.logaddexp(0, r) - y_bin * r)

    def fit(self, x, s, y):
        """Fit model to inputs, correct sides, and choices."""
        
        # Initial parameter valuesj
        theta_0 = np.array([0, 0, 0, 0])

        # Estimate parameters by minimizing log-likelihood
        res = opt.minimize(
            IBLModel.neg_LL, 
            theta_0, 
            args=(x, s, y), 
            method='BFGS', 
            options={'gtol':1e-2}
        )
        
        # Set parameter values
        self.logit_h = res.x[0]
        self.h = expit(self.logit_h)
        self.logit_a = res.x[1]
        self.a = expit(self.logit_a)
        self.w_0 = res.x[2]
        self.w_1 = res.x[3]

        # Store optimization result
        self.opt_result = res

    def decision_function(self, x, s):
        """Compute decision function (log-posterior ratio) for fit model."""

        r, _ = log_pos_full(x, s, self.logit_h, self.logit_a, self.w_0, self.w_1)

        return r

    def predict(self, x, s):
        """Predict choice for given inputs."""

        return np.sign(self.decision_function(x, s))

    def predict_proba(self, x, s):
        """Return choice probabilities (y=1) for given inputs."""

        return expit(self.decision_function(x, s))
