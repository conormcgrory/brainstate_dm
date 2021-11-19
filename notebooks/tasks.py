"""Functions for creating synthetic task data"""

import numpy as np

from numpy.random import default_rng


class GlazeTask:

    def __init__(self, hazard_rate, noise, s_init=1, rng=default_rng()):

        # Set parameters
        self.hazard_rate = hazard_rate
        self.noise = noise
        self.rng = rng

        # Initialize side
        self.s_current = s_init

    def sample_trials(self, n_trials):

        s_vals = np.full(n_trials, np.nan)
        x_vals = np.full(n_trials, np.nan)

        for i in range(n_trials):
        
            # Add current side and observation to arrays
            s_vals[i] = self.s_current
            x_vals[i] = self.s_current + self.rng.normal(0, self.noise)
        
            # Side switches with probability equal to hazard rate
            if self.rng.binomial(1, self.hazard_rate):
                self.s_current = -self.s_current

        return s_vals, x_vals
 

#def gen_glaze_trials(n_smps, rng, hazard_rate=0.1, noise=0.5, s_init=0):
#   
#    side = s_init
#    
#    for i in range(n_smps):
#        
#        x = side + rng.normal(0, noise)
#        
#        yield (side, x)
#        
#        # Switch sides with probability equal to hazard rate
#        if rng.binomial(1, hazard_rate):
#            side = -side
            
            
def gen_ibl_trials(n_smps, rng, hazard_rate=0.1, p_right=[0.2, 0.8], c_vals=[12, 25, 50, 100], b_init=0):
   
    block = b_init
    
    for i in range(n_smps):
        
        # Sample side (L=-1, R=1) using probability for current block value
        side = 1 - 2 * rng.binomial(1, p_right[block])
        
        # Sample contrast uniformly from c_vals, with sign determined by side
        signed_contrast = side * rng.choice(c_vals) 
        
        yield (block, side, signed_contrast)
        
        # Switch blocks with Probability equal to hazard rate
        if rng.binomial(1, hazard_rate):
            block = 1 - block
 
