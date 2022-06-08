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


@dataclass
class IBLFilterResult:
    """Structure for storing results of IBL model filter."""

    # Log- prior, likelihood, and posterior ratios for block
    b_prior: np.ndarray
    b_lik: np.ndarray
    b_pos: np.ndarray

    # Log- prior, likelihood, and posterior ratios for block
    s_prior: np.ndarray
    s_lik: np.ndarray
    s_pos: np.ndarray


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


def run_block_filter(s: np.ndarray, params: IBLParams):
    """Run Bayesian filtering on block variable."""

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

    return b_prior, b_lik, b_pos


def run_filter(s: np.ndarray, x: np.ndarray, 
        params: IBLParams) -> IBLFilterResult:
    """Recursively compute log-posterior ratios for all time points"""

    # Run filter on block variable
    b_prior, b_lik, b_pos = run_block_filter(s, params)
    
    # Use block prior and stimulus to compute posterior over side
    s_prior = phi(b_prior, params.beta)
    s_lik = params.bias + params.coef * x
    s_pos = s_lik + s_prior

    return IBLFilterResult(b_prior, b_lik, b_pos, s_prior, s_lik, s_pos)


def log_prior_side(s: np.ndarray, params: IBLParams) -> np.ndarray:
    """Compute prior over stimulus side."""

    # Run Bayesian filter on block
    b_prior, _, _ = run_block_filter(s, params)

    return phi(b_prior, params.beta)


def log_pos_side(s: np.ndarray, x: np.ndarray, params: IBLParams) -> np.ndarray:
    """Compute log-posterior ratio of stimulus sides for all time points."""

    res = run_filter(s, x, params)
    return res.s_pos


def nll_session(params: IBLParams, s :np.ndarray, 
        x: np.ndarray, y: np.ndarray) -> float:
    """Negative log-likelihood of single session given parameters."""
    
    # Compute log posterior ratio for stimulus side
    s_pos = log_pos_side(s, x, params)

    # Convert observed choices from [-1, 1] to [0, 1]
    y_bin = (y + 1) / 2
    
    return np.sum(np.logaddexp(0, s_pos) - y_bin * s_pos)


def nll(params: IBLParams, s: list[np.ndarray], 
        x: list[np.ndarray], y: list[np.ndarray]) -> float:
    """Negative log-likelihood of multiple sessions given parameters."""

    return sum(nll_session(params, s[i], x[i], y[i]) for i in range(len(s)))


def fit_ibl(s: list[np.ndarray], x: list[np.ndarray], 
        y: list[np.ndarray]) -> IBLParams:
    """Fit IBL model to multiple sessions."""
        
    # Initial parameter values in vector form
    params_0 = IBLParams(alpha=0, beta=0, bias=0, coef=0)
    pvec_0 = params_0.to_vec()

    # NLL function that takes parameters in vector form
    def nll_vec(pvec, s, x, y):
        return nll(IBLParams.from_vec(pvec), s, x, y)

    # Estimate parameters by minimizing log-likelihood (basinhopping
    # necessary for avoiding local minima)
    res = opt.basinhopping(
        nll_vec,
        pvec_0,
        minimizer_kwargs={
            'method': 'BFGS',
            'args': (s, x, y)
        },
        niter=10
    )

    return IBLParams.from_vec(res.x)


def sample_behavior(s: np.ndarray, x: np.ndarray, params: IBLParams, 
        rng=default_rng()) -> tuple[np.ndarray, IBLFilterResult]:
    """Sample choices from IBL agent for given input."""

    # Log posterior ratio for inputs
    f_result = run_filter(s, x, params)

    # Probability of choosing 1 (right)
    p = expit(f_result.s_pos)

    # Generate samples
    y = 2 * rng.binomial(1, p) - 1
        
    return y, f_result


def predict_choice(s: np.ndarray, x: np.ndarray, 
        params: IBLParams) -> np.ndarray:
    """Predict choice for given inputs."""

    return np.sign(log_pos_side(s, x, params))


def predict_proba(s: np.ndarray, x: np.ndarray, 
        params: IBLParams) -> np.ndarray:
    """Return choice probabilities (y=1) for given inputs."""

    return np.expit(log_pos_side(s, x, params))