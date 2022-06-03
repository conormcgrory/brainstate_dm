import numpy as np

def phi(a: float, b: float):
    """Function used to recursively compute prior term"""

    return np.logaddexp(0, a + b) - np.logaddexp(a, b)

def block_log_prior(q_prev, alpha):
    """Log-prior ratio for block"""

    return phi(q_prev, alpha)

def block_log_lik(side, beta):
    """Log-likelihood ratio for block given side value."""

    return side * beta

def side_log_prior(q_prev, alpha, beta):
    """Log-prior ratio for side given posterior over previous block"""

    return phi(phi(q_prev, alpha), beta)

def side_log_lik(x, bias, coef):
    """Log-likelihood ratio for side given single input"""
    
    return bias + coef * x

def log_pos_full(x, s, alpha, beta, bias, coef):
    """Recursively compute log-posterior ratios for all time points"""
    
    q = np.full_like(x, np.nan)
    r = np.full_like(x, np.nan)
        
    q[-1] = 0
    r[-1] = 0

    for t in range(x.shape[0]):
            
        # Use previous block prior and input to compute log-posterior over s_t
        r[t] = side_log_lik(x[t], bias, coef) + side_log_prior(q[t - 1], alpha, beta)
            
        # After observing s[t] in feedback, update log-posterior over block
        q[t] = block_log_lik(s[t], beta) + block_log_prior(q[t - 1], alpha)
        

    return r, q