"""Bayesian filtering for IBL task"""

from dataclasses import dataclass

import numpy as np
import scipy.optimize as opt
from numpy.random import default_rng
from scipy.special import logit, expit

from bfdm.ibldata import Session


@dataclass
class IBLParams:
    """Parameters for IBL model."""

    # Logit of "stay rate" (1 - hazard rate)
    alpha: float

    # Logit of probability that side is same as block
    beta: float

    # Bias term for log-likelihood
    bias: float

    # Coefficient term for log-likelihood
    coef: float

    def hazard_rate(self):
        """Probability of block switching at any time point."""

        return expit(-self.alpha)

    def p_stay(self):
        """Probability of block staying the same at any time point."""

        return expit(self.alpha)

    def p_side(self):
        """Probability that side is same as block."""

        return expit(self.beta)

    def to_vec(self):
        """Convert parameters to vector (used for optimization functions)."""

        return np.array([self.alpha, self.beta, self.bias, self.coef])
    
    @classmethod
    def from_vec(cls, pvec):
        """Extract parameters from vector (used for optimization functions)"""

        return cls(alpha=pvec[0], beta=pvec[1], bias=pvec[2], coef=pvec[3])


def get_optimal_params(p_stay: float, p_side: float, noise: float) -> IBLParams:
    """Return optimal model parameters based on task parameters."""

    alpha = logit(p_stay)
    beta = logit(p_side)
    bias = 0
    coef = 2 / (noise ** 2)

    return IBLParams(alpha, beta, bias, coef)


def phi(a, b):
    """Function used to recursively compute prior term"""

    return np.logaddexp(0, a + b) - np.logaddexp(a, b)


def run_filter(x: np.ndarray, s: np.ndarray, params: IBLParams):
    """Recursively compute log-posterior ratios for all time points"""
    
    q = np.full_like(x, np.nan)
    r = np.full_like(x, np.nan)
    side_log_prior = np.full_like(x, np.nan)
        
    q[-1] = 0

    for t in range(x.shape[0]):
            
        # Use previous block prior and input to compute log-posterior over s_t
        block_log_prior = phi(q[t - 1], params.alpha)
        side_log_prior[t] = phi(block_log_prior, params.beta)
        side_log_lik = params.bias + params.coef * x[t] 
        r[t] = side_log_lik + side_log_prior[t]
            
        # After observing s[t] in feedback, update log-posterior over block
        block_log_lik = s[t] * params.beta
        q[t] = block_log_lik + block_log_prior

    return r, side_log_prior


def log_pos_side(x: np.ndarray, s: np.ndarray, params: IBLParams):
    """Compute log-posterior ratio of stimulus sides for all time points."""

    return run_filter(x, s, params)[0]


def log_prior_side(s: np.ndarray, params: IBLParams):

    # Logit of prior, likelihood, and posterior for block value
    b_prior = np.full_like(s, np.nan)
    b_lik = np.full_like(s, np.nan)
    b_pos = np.full_like(s, np.nan)
       
    # Run Bayesian filter on block variable
    b_pos[-1] = 0
    for t in range(s.shape[0]):
        b_prior[t] = phi(b_pos[t - 1], params.alpha)
        b_lik[t] = s[t] * params.beta
        b_pos[t] = b_lik[t] + b_prior[t]

    # Use prior over block to compute prior over stimulus side s[t]
    s_prior = phi(b_prior, params.beta)

    return s_prior


def neg_LL_session(params: IBLParams, x, s, y):
    """Negative log-likelihood of single session given parameters."""
    
    r = log_pos_side(x, s, params)
    y_bin = (y + 1) / 2
    
    return np.sum(np.logaddexp(0, r) - y_bin * r)

def neg_LL(params: IBLParams, sessions: list[Session]):
    """Negative log-likelihood of multiple sessions given parameters."""

    nll_vals = np.array([neg_LL_session(params, s.x, s.s, s.y) for s in sessions])

    return np.sum(nll_vals)

def fit_ibl_session(x, s, y):
    """Fit IBL model to single session."""
        
    # Initial parameter values in vector form
    params_0 = IBLParams(alpha=0, beta=0, bias=0, coef=0)
    pvec_0 = params_0.to_vec()

    # NLL function that takes parameters in vector form
    def nll_vec(pvec, xx, ss, yy):
        return neg_LL_session(IBLParams.from_vec(pvec), xx, ss, yy)

    # Estimate parameters by minimizing log-likelihood (basinhopping
    # necessary for avoiding local minima)
    res = opt.basinhopping(
        nll_vec,
        pvec_0, 
        minimizer_kwargs={
            'method': 'BFGS',
            'args': (x, s, y)
        },
        niter=10
    )

    return IBLParams.from_vec(res.x)

def fit_ibl(sessions: list[Session]):
    """Fit IBL model to multiple sessions."""
        
    # Initial parameter values in vector form
    params_0 = IBLParams(alpha=0, beta=0, bias=0, coef=0)
    pvec_0 = params_0.to_vec()

    # NLL function that takes parameters in vector form
    def nll_vec(pvec, sdata):
        return neg_LL(IBLParams.from_vec(pvec), sdata)

    # Estimate parameters by minimizing log-likelihood (basinhopping
    # necessary for avoiding local minima)
    res = opt.basinhopping(
        nll_vec,
        pvec_0,
        minimizer_kwargs={
            'method': 'BFGS',
            'args': (sessions)
        },
        niter=10
    )

    return IBLParams.from_vec(res.x)


def sample_behavior(x: np.ndarray, s: np.ndarray, 
        params: IBLParams, return_rp=False, rng=default_rng()):
    """Sample choices from IBL agent for given input."""

    # Log posterior ratio for inputs
    r, log_prior = run_filter(x, s, params)

    # Probability of choosing 1 (right)
    p = expit(r)

    # Generate samples
    y = 2 * rng.binomial(1, p) - 1
        
    if return_rp:
        return y, r, log_prior
    else:
        return y


def predict_choice(x: np.ndarray, s: np.ndarray, params: IBLParams) -> np.ndarray:
    """Predict choice for given inputs."""

    return np.sign(log_pos_side(x, s, params))


def predict_proba(x: np.ndarray, s: np.ndarray, params: IBLParams) -> np.ndarray:
    """Return choice probabilities (y=1) for given inputs."""

    return np.expit(log_pos_side(x, s, params))