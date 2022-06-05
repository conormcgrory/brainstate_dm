"""Fit IBL model to all unbiased blocks."""

import numpy as np
import pandas as pd
import datajoint as dj

from bfdm.ibldata import get_session_data
from bfdm.iblmodel import fit_ibl


SESSION_LIST_FPATH = '../data/ibl/trained_sessions.csv'


def main():

    # Connect to IBL database via DataJoint
    dj.config['database.host'] = 'datajoint-public.internationalbrainlab.org'
    dj.config['database.user'] = 'ibldemo'
    dj.config['database.password'] = 'sfn2019demo'
    dj.conn()

    # Load list of selected sessions
    sessions_df = pd.read_csv(SESSION_LIST_FPATH, index_col=0)

    # List for holding extracted session data
    sessions = []

    for row in sessions_df.itertuples():

        try:

            # Download data for session
            df = get_session_data(row.subject_uuid, row.session_start_time) 

            # Select rows from unbiased block at beginning of session
            df = df[df.block == 0]

            # Load data from DataFrame into numpy array
            data = df[['correct_side', 'signed_contrast', 'choice']].to_numpy()
            s = data[:, 0]
            x = data[:, 1]
            y = data[:, 2]

            # If first 90 unbiased trials are all present, add to list
            if x.shape[0] == 90:
                sessions.append((x, s, y))

        except KeyError as e:

            print('')
            print(f'KeyError! Missing required field: {str(e)}')
            print(f'subject_uuid:{row.subject_uuid}')
            print(f'session_start_time:{row.session_start_time}')
            print('')

    # Fit IBL model to data
    params = fit_ibl(sessions)

    print(params)


if __name__ == '__main__':
    main()
