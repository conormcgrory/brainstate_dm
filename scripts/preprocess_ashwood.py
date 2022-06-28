"""Script for selecting and preprocessing data using Ashwood criteria."""

from collections import defaultdict

import numpy as np
from one.api import ONE, One
from tqdm import tqdm

from bfdm import ibldata


# Local directory where example IBL data is stored 
RAW_DATA_DIR = 'data/raw/ibl_example'

# Local directory where processed data is stored
PROCESSED_DATA_DIR = 'data/processed/ashwood'

# Number of sessions needed to include subject in dataset
N_REQ_SESSIONS = 30


def get_all_session_ids(one: One) -> tuple[list[str], list[str]]:
    """Query database for list of all session EIDs and corresponding subject IDs."""

    # Query all sessions in ibl.trials table
    eids, info = one.search(['_ibl_trials.*'])

    # Extract subject ID from info dict
    subjects = [x['subject'] for x in info]

    return eids, subjects


def include_session(rdata: ibldata.RawSessionData) -> bool:
    """Criteria for including session in dataset."""

    # Check that session has unbiased (p=0.5) and biased (p=0.2, p=0.8) blocks 
    pvals_data = np.unique(rdata.probability_left)
    pvals_expected = np.array([0.2, 0.5, 0.8])
    if not np.array_equal(pvals_data, pvals_expected):
        return False

    # Check that unbiased block doesn't have any 'NO-GO' (0) choice values
    idx_unbiased = np.where(rdata.probability_left == 0.5)[0]
    cvals_data = np.unique(rdata.choice[idx_unbiased])
    cvals_expected = np.array([-1, 1])
    if not np.array_equal(cvals_data, cvals_expected):
        return False
        
    return True


def main():

    # Connect to local database
    one = ONE(mode='local', cache_dir=RAW_DATA_DIR)

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
            sdata = ibldata.get_processed_unbiased_data(rdata)
            data[subject].append(sdata)

    print('Done.')

    # Delete all subjects with fewer than 30 sessions
    data = {s: x for (s, x) in data.items() if len(x) >= N_REQ_SESSIONS}

    # Save data to processed data directory 
    print(f'Saving data to {PROCESSED_DATA_DIR}...')
    if data:
        ibldata.save_sessions(data, PROCESSED_DATA_DIR)
    else:
        raise ValueError('No valid sessions found!')
    print('Done.')



if __name__ == '__main__':
    main()
