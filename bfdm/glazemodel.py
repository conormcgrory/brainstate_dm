"""Bayesian filtering model from Glaze et al., 2015"""

import numpy as np
import scipy.optimize as opt
from dataclasses import dataclass

from numpy.random import default_rng
from scipy.special import logit, expit


@dataclass
class GlazeParams:
    """Parameters for Glaze model."""

    # Logit of "stay rate" (1 - hazard rate)
    alpha: float

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

    def to_vec(self):
        """Convert parameters to vector (used for optimization functions)."""

        return np.array([self.alpha, self.bias, self.coef])
    
    @classmethod
    def from_vec(cls, pvec):
        """Extract parameters from vector (used for optimization functions)"""

        return cls(alpha=pvec[0], bias=pvec[1], coef=pvec[2])


@dataclass
class GlazeFilterResult:
    """Structure for storing results of Glaze model filter."""

    # Log prior ratio for stimulus side
    s_prior: np.ndarray

    # Log likelihood ratio for stimulus side
    s_lik: np.ndarray

    # Log posterior ratio for stimulus side
    s_pos: np.ndarray


def get_optimal_params(p_stay: float, noise: float) -> GlazeParams:
    """Return optimal model parameters based on task parameters."""

    alpha = logit(p_stay)
    bias = 0
    coef = 2 / (noise ** 2)

    return GlazeParams(alpha, bias, coef)


def phi(a, b):
    """Function used to recursively compute prior term"""

    return np.logaddexp(0, a + b) - np.logaddexp(a, b)


def run_filter(x: np.ndarray, params: GlazeParams) -> GlazeFilterResult:
    """Run Glaze filtering model and return log-posterior ratio over side."""
    
    s_prior = np.full_like(x, np.nan)
    s_lik = np.full_like(x, np.nan)
    s_pos = np.full_like(x, np.nan)
        
    s_pos[-1] = 0

    for t in range(x.shape[0]):
        s_prior[t] = phi(s_pos[t - 1], params.alpha)
        s_lik[t] = params.bias + params.coef * x[t]
        s_pos[t] = s_lik[t] + s_prior[t]
            
    return GlazeFilterResult(s_prior, s_lik, s_pos)


def log_pos_side(x: np.ndarray, params: GlazeParams) -> np.ndarray:
    """Return log posterior ratio for stimulus side based on filter."""

    res = run_filter(x, params)
    return res.s_pos


def nll_session(params: GlazeParams, x: np.ndarray, y: np.ndarray):
    """Negative log-likelihood of single session given parameters."""
    
    # Compute log posterior ratio for stimulus side
    s_pos = log_pos_side(x, params)

    # Convert observed choices from [-1, 1] to [0, 1]
    y_bin = (y + 1) / 2
    
    return np.sum(np.logaddexp(0, s_pos) - y_bin * s_pos)


def nll(params: GlazeParams, x: list[np.ndarray], y: list[np.ndarray]) -> float:
    """Negative log-likelihood of multiple sessions given parameters."""

    return sum(nll_session(params, x[i], y[i]) for i in range(len(x)))


def fit_glaze(x: list[np.ndarray], y: list[np.ndarray]) -> GlazeParams:
    """Fit Glaze model to multiple sessions."""
        
    # Initial parameter values in vector form
    params_0 = GlazeParams(alpha=0, bias=0, coef=0)
    pvec_0 = params_0.to_vec()

    # NLL function that takes parameters in vector form
    def nll_vec(pvec, x, y):
        return nll(GlazeParams.from_vec(pvec), x, y)

    # Estimate parameters by minimizing log-likelihood (basinhopping
    # necessary for avoiding local minima)
    res = opt.basinhopping(
        nll_vec,
        pvec_0,
        minimizer_kwargs={
            'method': 'BFGS',
            'args': (x, y)
        },
        niter=10
    )

    return GlazeParams.from_vec(res.x)


def sample_behavior(x: np.ndarray, params: GlazeParams, 
        rng=default_rng()) -> tuple[np.ndarray, GlazeFilterResult]:
    """Sample choices from Glaze model."""

    # Run filter on input
    f_result= run_filter(x, params)

    # Probability of choosing 1 (right)
    p = expit(f_result.s_pos)

    # Generate samples
    y = 2 * rng.binomial(1, p) - 1

    return y, f_result


def predict_choice(x: np.ndarray, params: GlazeParams) -> np.ndarray:
    """Predict choice for given inputs."""

    return np.sign(log_pos_side(x, params))


def predict_proba(x: np.ndarray, params: GlazeParams) -> np.ndarray:
    """Return choice probabilities (y=1) for given inputs."""

    return np.expit(log_pos_side(x, params))