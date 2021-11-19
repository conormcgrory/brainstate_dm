"""Models for decision-making tasks"""

import numpy as np
import scipy.optimize as opt

from numpy.random import default_rng
from scipy.special import logit, expit


class GlazeModel:
    
    def __init__(self, rng=default_rng()):
        
        # Random number generator (for sampling)
        self.rng = rng
        
        # Logit of hazard rate
        self.z_hazard = None
        
        # Bias term
        self.w_0 = None
        
        # Weight for contrast
        self.w_1 = None
        
        # Result from optimization
        self.opt_result = None
            
    @property
    def hazard_rate(self):
        return expit(self.z_hazard)
    
    @staticmethod
    def phi(a, b):
        return np.logaddexp(0, a + b) - np.logaddexp(a, b)

    @staticmethod
    def compute_log_posterior(x, z_hazard, w_0, w_1):
    
        r = np.full_like(x, np.nan)
        r[-1] = 0

        for t in range(x.shape[0]):
            r[t] = w_0 + w_1 * x[t] + GlazeModel.phi(r[t - 1], -z_hazard)
        
        return r

    @staticmethod
    def neg_LL(p, x, y):
    
        r = GlazeModel.compute_log_posterior(x, p[0], p[1], p[2])
        y_bin = (y + 1) / 2
    
        return np.sum(np.logaddexp(0, r) - y_bin * r)

    def fit(self, x, y):
        
        p_0 = np.array([0, 0, 0])
        res = opt.minimize(GlazeModel.neg_LL, p_0, args=(x, y), method='BFGS', options={'gtol':1e-2})
        
        self.opt_result = res
        self.z_hazard = res.x[0]
        self.w_0 = res.x[1]
        self.w_1 = res.x[2]
        
    def log_posterior_ratio(self, x):
        
        return GlazeModel.compute_log_posterior(x, self.z_hazard, self.w_0, self.w_1)
        
    def sample(self, x):
        
        r = GlazeModel.compute_log_posterior(x, self.z_hazard, self.w_0, self.w_1)
        p = expit(r)
       
        return self.rng.binomial(1, p)
        
        
        
        
class IBLModel:
    
    def __init__(self):
        
        self.z_hazard = None
        self.z_side = None
        self.w_0 = None
        self.w_1 = None
        
        self.opt_result = None
        
    @property
    def hazard_rate(self):
        return expit(self.z_hazard)
    
    @property
    def p_side(self):
        return expit(self.z_side)

    @staticmethod
    def phi(a, b):
        return np.logaddexp(0, a + b) - np.logaddexp(a, b)
   
    @staticmethod
    def block_LP(r_prev, z_hazard):
        return phi(r_prev, z_hazard)

    @staticmethod
    def block_LL(side, z_side):
        return side * z_side

    @staticmethod
    def side_LP(b_prev, z_hazard, z_side):
        return phi(b_prev, phi(z_hazard, z_side))

    @staticmethod
    def side_LL(signed_contrast, w_0, w_1):
        return w_0 + w_1 * signed_contrast

    @staticmethod
    def compute_log_posterior(x, s, z_hazard, z_side, w_0, w_1, return_b=False):
    
        r = np.full_like(x, np.nan)
        b = np.full_like(x, np.nan)
        
        r[-1] = 0
        b[-1] = 0

        for t in range(x.shape[0]):
            
            # Use previous block log-odds and contrast to compute side log-odds
            r[t] = IBLModel.side_LL(x[t], w_0, w_1) + IBLModel.side_LP(b[t - 1], z_hazard, z_side)
            
            # Use feedback to update block log-odds
            b[t] = IBLModel.block_LL(s[t], z_side) + IBLModel.block_LP(r[t - 1], z_hazard)
        
        if return_b:
            return r, b
        else:
            return r

    @staticmethod
    def neg_LL(p, x, s, y):
    
        r = IBLModel.compute_log_posterior(x, s, p[0], p[1], p[2], p[3])
        y_bin = (y + 1) / 2
    
        return np.sum(np.logaddexp(0, r) - y_bin * r)

    def fit(self, x, s, y):
        
        p_0 = np.array([0, 0, 0, 0])
        
        res = opt.minimize(IBLModel.neg_LL, p_0, args=(x, y, s), method='BFGS', options={'gtol':1e-2})
        
        self.opt_result = res
        self.z_hazard = res.x[0]
        self.z_side = res.x[1]
        self.w_0 = res.x[2]
        self.w_1 = res.x[3]
        
    def predict(self, x, s, return_posterior=False):
        
        r, b = IBLModel.compute_log_posterior(
            x, s, self.z_hazard, self.z_side, self.w_0, self.w_1, return_b=True)
        
        y_pred = np.sign(r)
        
        if return_posterior:
            return y_pred, r, b
        else:
            return y_pred
