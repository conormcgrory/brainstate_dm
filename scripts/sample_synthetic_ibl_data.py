"""Script for sampling synthetic data from IBL model."""

import numpy as np
from numpy.random import default_rng
from collections import defaultdict

from bfdm import ibldata, ibltask, iblmodel

# Output directory
OUT_DIRPATH = 'data/synthetic/biased_ibl'

# Number of subjects, sesssions, and trials per session
N_SUBJECTS = 20
N_SESSIONS = 45
N_TRIALS = 850

# Task parameters
P_STAY = 0.98
P_SIDE = 0.8
NOISE = 1.0


def sample_session(eid, model_params, rng):

    # Sample task data
    block, side, contrast = ibltask.sample_biased_trials(
        N_TRIALS, P_STAY, P_SIDE, NOISE, rng)

    # Sample choice behavior
    choice, _ = iblmodel.sample_behavior(side, contrast, model_params, rng)

    return ibldata.SessionData(
        eid=eid,
        block=block,
        side=side,
        contrast=contrast,
        choice=choice
    )


def main():

    rng = default_rng()

    # Use optimal model parameters for task
    model_params = iblmodel.get_optimal_params(P_STAY, P_SIDE, NOISE)

    data = defaultdict(list)
    for i in range(N_SUBJECTS):
        for j in range(N_SESSIONS):
            subject = f'fakemouse_{i:02}'
            eid = f'{i:02}{j:02}'
            sdata = sample_session(eid, model_params, rng)
            data[subject].append(sdata)

    ibldata.save_sessions(data, OUT_DIRPATH)


if __name__ == '__main__':
    main()
