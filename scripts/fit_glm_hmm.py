"""Fit GLM-HMM model to all sessions.

The goal of this script is to replicate the results from Ashwood et al.

"""

import argparse
import os
import json

import numpy as np
import ssm

from bfdm import ibldata

# Scale factor used for contrast values, currently set to reciprocal of standard
# deviation of contrast values (Because constrast values are uniform 
# distribution over +-[0.0, 0.0625, 0.125, 0.25, 1.0], standard deviation is
# ~0.4652, which makes scale factor ~2.15
SCL_FACTOR = 2.15


def parse_args():

    parser = argparse.ArgumentParser(
        description='fit GLM-HMM model to preprocessed IBL data'
    )
    parser.add_argument(
        '-i', '--input', type=str, required=True,
        help='path to preprocessed IBL data'
    )
    parser.add_argument(
        '-o', '--output', type=str, required=True,
        help='path where results are stored'
    )
    parser.add_argument(
        '-n', '--nstates', type=int, required=True,
        help='number of HMM states'
    )

    return parser.parse_args()


def get_model_inputs(x: np.ndarray) -> np.ndarray:
    """Modify inputs for GLM-HMM (need to add column of ones)."""

    # Scale contrast values by dividing by standard deviation
    x_scl = x * SCL_FACTOR

    x_mod = np.ones((x.shape[0], 2))
    x_mod[:, 0] = x_scl

    return x_mod


def get_model_choices(y: np.ndarray) -> np.ndarray:
    """Modify choices for GLM-HMM (y=-1 maps to y=1 and y=1 maps to y=0)

    This is done because the GLM-HMM model code in the `ssm` package takes
    choice values from [0, 1] instead of [-1, 1], and uses its inputs to 
    compute p(y=0). Because large, positive contrast values increase the 
    probability that the choice is right (+1), the right choice (+1) is encoded
    here as 0, and the left choice (-1) as +1.
    
    """

    y_mod = np.reshape(y, (-1, 1))
    y_mod = (1 - y_mod) / 2
    y_mod = y_mod.astype(np.int64)

    return y_mod


def params_to_dict(params) -> dict:
    """Convert fit parameters from GLM-HMM model to dict format."""

    pdict = {}
    for i in range(len(params)):
        k = f'state {i}'
        v = dict(bias=params[i][0][1], coef=params[i][0][0])
        pdict[k] = v

    return pdict


def main():

    # Parse command-line arguments
    args = parse_args()
    if not os.path.exists(args.input):
        print(f'Error: Input directory {args.input} does not exist.')
        return -1
    if os.path.exists(args.output):
        print(f'Error: Output path {args.output} already exists.')
        return -2
    if args.nstates < 1:
        print(f'Error: nstates needs to be a positive integer.')
        return -3

    print('Loading unbiased session data...')
    data_dict = ibldata.load_sessions(args.input)
    sessions = [s for s_list in data_dict.values() for s in s_list]
    n_sessions = len(sessions)
    print(f'Done. Loaded {n_sessions} sessions.')

    print('Fitting GLM-HMM model...')

    # Convert inputs and choices to GLM-HMM format
    inputs = [get_model_inputs(s.contrast) for s in sessions]
    choices = [get_model_choices(s.choice) for s in sessions]

    glm_hmm = ssm.HMM(
        args.nstates, 1, 2, 
        observations="input_driven_obs", 
        observation_kwargs=dict(C=2),
        transitions="standard"
    )
    fit_loglik = glm_hmm.fit(
        choices, 
        inputs=inputs, 
        method="em", 
        num_iters=300, 
        tolerance=1e-4
    )
    params = glm_hmm.observations.params

    print('Done.')

    print('Saving results...')
    params_dict = params_to_dict(params)
    results_dict = {
        'input_dpath': args.input,
        'n_sessions': n_sessions,
        'scl_factor': SCL_FACTOR,
        'n_states': args.nstates,
        'params': params_dict,
        'fit_loglik': fit_loglik,
    }
    with open(args.output, 'w') as f:
        json.dump(results_dict, f)
    print('Done.')

    print('Parameters:')
    print(json.dumps(params_dict, indent=4))


if __name__ == '__main__':
    main()