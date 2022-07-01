"""Script for sampling synthetic unbiased data."""

import numpy as np
from numpy.random import default_rng
from scipy.special import logit, expit
from collections import defaultdict

from bfdm import ibldata

# Output directory
OUT_DIRPATH = 'data/synthetic/unbiased_glm'

# Number of subjects, sesssions, and trials per session
N_SUBJECTS = 5
N_SESSIONS = 30
N_TRIALS = 90

# Coefficient and bias parameters of ground-truth GLM
COEF = 1.0
BIAS = -0.5


def sample_session(eid, decision_fn, rng):

    # Block (always zero)
    block = np.zeros(N_TRIALS)

    # Stimulus side
    side = 2 * rng.binomial(1, 0.5, N_TRIALS) - 1

    # Signed contrast
    cvals = np.array([0.0, 0.0625, 0.125, 0.25, 1.0])
    contrast = side * rng.choice(cvals, size=N_TRIALS)

    # Compute decision function and sample choices
    p_right = decision_fn(contrast)
    choice = 2 * rng.binomial(1, p_right) - 1

    return ibldata.SessionData(
        eid=eid,
        block=block,
        side=side,
        contrast=contrast,
        choice=choice
    )


def logistic_decision_fn(contrast):

    return expit(COEF * contrast + BIAS)


def main():

    rng = default_rng()

    data = defaultdict(list)
    for i in range(N_SUBJECTS):
        for j in range(N_SESSIONS):
            subject = f'fakemouse_{i:02}'
            eid = f'{i:02}{j:02}'
            sdata = sample_session(eid, logistic_decision_fn, rng)
            data[subject].append(sdata)

    ibldata.save_sessions(data, OUT_DIRPATH)


if __name__ == '__main__':
    main()
