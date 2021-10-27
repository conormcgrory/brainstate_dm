"""Utility functions"""

def gen_glaze_trials(n_smps, rng, hazard_rate=0.1, noise=0.5, s_init=0):
   
    side = s_init
    
    for i in range(n_smps):
        
        x = side + rng.normal(0, noise)
        
        yield (side, x)
        
        # Switch sides with probability equal to hazard rate
        if rng.binomial(1, hazard_rate):
            side = -side
            
            
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
            
            
def smp_glaze_trials(n_smps, rng, hazard_rate=0.1, noise=0.5, s_init=1):
    
    changes = rng.binomial(1, hazard_rate, n_smps)
    s_right = np.mod(np.cumsum(changes), 2)
    s_smps = s_init * (2 * s_right - 1)
    x_smps = s_smps + rng.normal(0, noise, n_smps)
    
    return s_smps, x_smps