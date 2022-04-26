"""Run tests on all selected sessions to ensure validity of analysis."""

import numpy as np
import pandas as pd
import datajoint as dj

from ibldata import get_session_data


SESSION_LIST_FPATH = '../data/ibl/trained_sessions.csv'


def main():

    # Connect to IBL database via DataJoint
    dj.config['database.host'] = 'datajoint-public.internationalbrainlab.org'
    dj.config['database.user'] = 'ibldemo'
    dj.config['database.password'] = 'sfn2019demo'
    dj.conn()

    # Load list of selected sessions
    sessions_df = pd.read_csv(SESSION_LIST_FPATH, index_col=0)

    for row in sessions_df.itertuples():

        # Download data for session
        df = get_session_data(row.subject_uuid, row.session_start_time)

        # Check that all side probabilities are either 0.5, 0.2, or 0.8
        p_vals = df.trial_stim_prob_left.unique()
        if set(p_vals) != set([0.5, 0.2, 0.8]):
            print('Incorrect task!')
            print(f'uuid: {row.subject_uuid}')
            print(f'time: {row.session_start_time}')
            print(f'p_vals: {p_vals}')

        # TODO: Add additional checks here


    print('Done.')


if __name__ == '__main__':
    main()
