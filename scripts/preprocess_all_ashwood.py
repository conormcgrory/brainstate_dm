"""Select and preprocess all blocks of sessions (Ashwood criteria).

This script takes two arguments: the path to the local ONE database where the
raw data is stored (-i or --input), and the path to where the preprocessed data
will be stored (-o or --output). For example, to extract preprocessed sessions
from a database stored at `data/foo` and store the results at `data/bar`, the 
command would be:
```console
> python scripts/preprocess_all_ashwood -i data/foo -o data/bar
```
If the input and output arguments are not given, the script will use default
values for these arguments.
"""

import argparse
import os
from collections import defaultdict

import numpy as np
from one.api import ONE, One
from tqdm import tqdm

from bfdm import ibldata


# Number of sessions needed to include subject in dataset
N_REQ_SESSIONS = 30

# Default location of ONE database containing raw data
DEFAULT_INPUT_DIR = 'data/raw/ibl_example'

# Default output directory path
DEFAULT_OUTPUT_DIR = 'data/preprocessed/ashwood_all'


def parse_args():

    parser = argparse.ArgumentParser(
        description='select and preprocess all blocks (Ashwood criteria).'
    )
    parser.add_argument(
        '-i', '--input', type=str,
        default=DEFAULT_INPUT_DIR,
        help='path to local ONE database where raw data is stored.'
    )
    parser.add_argument(
        '-o', '--output', type=str,
        default=DEFAULT_OUTPUT_DIR,
        help='path where preprocessed data is stored.'
    )

    return parser.parse_args()


def get_all_session_ids(one: One) -> tuple[list[str], list[str]]:
    """Query database for list of all session EIDs and corresponding subject IDs."""

    # Query all sessions in ibl.trials table
    eids, info = one.search(['_ibl_trials.*'])

    # Extract subject ID from info dict
    subjects = [x['subject'] for x in info]

    return eids, subjects


def include_session(rdata: ibldata.RawSessionData) -> bool:
    """Criteria for including session in dataset (from Ashwood et al., 2022).

    The two critera used in the Ashwood et al. paper to determine if a session
    should be included are:
        1) The session needs to contain unbiased (p=0.5), left-biased (p=0.8),
        and right-biased (p=0.2) blocks.
        2) In the unbiased block (the only data used in the Ashwood paper), 
        there must be fewer than 10 'no-go' trials, where the animal doesn't 
        move the wheel.
    The inclusion criteria implemented here are almost exactly the same as those
    in the Ashwood paper. The only difference is that, instead of requiring that
    the animal has <10 'no-go' trials in the unbiased block, we require that it 
    has zero 'no-go' trials in all blocks (unbiased, left-biased, and 
    right-biased).
    """

    # Check that session has unbiased (p=0.5) and biased (p=0.2, p=0.8) blocks 
    pvals_data = np.unique(rdata.probability_left)
    pvals_expected = np.array([0.2, 0.5, 0.8])
    if not np.array_equal(pvals_data, pvals_expected):
        return False

    # Check that session doesn't have any 'NO-GO' (0) choice values
    cvals_data = np.unique(rdata.choice)
    cvals_expected = np.array([-1, 1])
    if not np.array_equal(cvals_data, cvals_expected):
        return False
        
    return True


def main():

    # Parse program arguments
    args = parse_args()
    if not os.path.exists(args.input):
        print(f'Error: Input directory {args.input} does not exist.')
        return -1
    if os.path.exists(args.output):
        print(f'Error: Output directory {args.output} already exists.')
        return -2

    # Connect to local database
    one = ONE(mode='local', cache_dir=args.input)

    # Get list of EIDs of all sessions in database, along with subject IDs
    print('Fetching session IDs...')
    eids, subjects = get_all_session_ids(one)
    print(f'Done {len(eids)} sessions found.')

    print('Fetching session data...')

    # Data dict maps subject ID to list of SessionData objects
    data = defaultdict(list)

    for eid, subject in tqdm(zip(eids, subjects), total=len(eids)):

        # Get raw session data from database
        rdata = ibldata.get_raw_session_data(eid, one)

        # If session is eligible, add to data dict
        if include_session(rdata):
            sdata = ibldata.get_processed_data(rdata)
            data[subject].append(sdata)

    print('Done.')

    # Delete all subjects with fewer than 30 sessions
    data = {s: x for (s, x) in data.items() if len(x) >= N_REQ_SESSIONS}

    # Save data to processed data directory 
    print(f'Saving data to {args.output}...')
    if data:
        ibldata.save_sessions(data, args.output)
    else:
        raise ValueError('No valid sessions found!')
    print('Done.')


if __name__ == '__main__':
    main()
