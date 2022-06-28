"""Fit GLM-HMM model to all unbiased sessions.

The goal of this script is just to replicate the results from Ashwood et al.

"""

import json

import numpy as np
import ssm

from bfdm import ibldata


# Local directory where processed data is stored
PROCESSED_DATA_DIR = 'data/processed/ashwood_test'

# Path to store fit parameters to
OUTPUT_FPATH = 'params_test.json'

# Number of states to use for GLM-HMM model
NUM_STATES = 3


def get_model_inputs(x):
    """Modify inputs for GLM-HMM (need to add column of ones)."""

    x_mod = np.ones((x.shape[0], 2))
    x_mod[:, 0] = x

    return x_mod


def get_model_choices(y):
    """Modify choices for GLM-HMM (y=1 maps to y=0 and y=-1 maps to y=1)."""

    y_mod = np.reshape(y, (-1, 1))
    y_mod = (1 - y_mod) / 2
    y_mod = y_mod.astype(np.int64)

    return y_mod


def params_to_dict(params):
    """Convert fit parameters from GLM-HMM model to dict format."""

    pdict = {}
    for i in range(len(params)):
        k = f'state {i}'
        v = dict(bias=params[i][0][0], coef=params[i][0][1])
        pdict[k] = v

    return pdict


def main():

    print('Loading unbiased session data...')
    data_dict = ibldata.load_sessions(PROCESSED_DATA_DIR)
    sessions = [s for s_list in data_dict.values() for s in s_list]
    print(f'Done. Loaded {len(sessions)} sessions.')

    print('Fitting GLM-HMM model...')

    # Convert inputs and choices to GLM-HMM format
    inputs = [get_model_inputs(s.contrast) for s in sessions]
    choices = [get_model_choices(s.choice) for s in sessions]

    glm_hmm = ssm.HMM(
        NUM_STATES, 1, 2, 
        observations="input_driven_obs", 
        observation_kwargs=dict(C=2),
        transitions="standard"
    )
    glm_hmm.fit(
        choices, 
        inputs=inputs, 
        method="em", 
        num_iters=200, 
        tolerance=1e-4
    )
    params = glm_hmm.observations.params

    print('Done.')

    print('Saving parameters...')
    pdict = params_to_dict(params)
    with open(OUTPUT_FPATH, 'w') as f:
        json.dump(pdict, f)
    print('Done.')

    print('Parameters:')
    print(json.dumps(pdict, indent=4))


if __name__ == '__main__':
    main()
