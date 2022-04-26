"""Fit IBL model to all sessions."""

from itertools import islice

import numpy as np
import pandas as pd
import datajoint as dj

from ibldata import get_session_data
from iblmodel import IBLModel


SESSION_LIST_FPATH = '../data/ibl/trained_sessions.csv'
PARAMS_FPATH = '../data/ibl/ibl_params.csv'
N_SESSIONS = 1000


def main():

    # Connect to IBL database via DataJoint
    dj.config['database.host'] = 'datajoint-public.internationalbrainlab.org'
    dj.config['database.user'] = 'ibldemo'
    dj.config['database.password'] = 'sfn2019demo'
    dj.conn()

    # Load list of selected sessions
    sessions_df = pd.read_csv(SESSION_LIST_FPATH, index_col=0)

    # Dict for holding model parameters
    params = {'h': [], 'a': [], 'w_0': [], 'w_1': []}

    for row in sessions_df.itertuples():

        try:

            # Download data for session
            df = get_session_data(row.subject_uuid, row.session_start_time) 

            # Load data from DataFrame into numpy array
            data = df[['correct_side', 'signed_contrast', 'choice']].to_numpy()
            s = data[:, 0]
            x = data[:, 1]
            y = data[:, 2]

            # Fit IBL model
            model = IBLModel()
            model.fit(x, s, y)

            # Add params to dict
            params['h'].append(model.h)
            params['a'].append(model.a)
            params['w_0'].append(model.w_0)
            params['w_1'].append(model.w_1)
    
            # Print results
            print(f'subject_uuid:{row.subject_uuid}')
            print(f'session_start_time:{row.session_start_time}')
            print(f'h: {model.h}')
            print(f'a: {model.a}')
            print(f'w_0: {model.w_0}')
            print(f'w_1: {model.w_1}') 
            print('')

        except KeyError as e:

            print('')
            print(f'KeyError! Missing required field: {str(e)}')
            print(f'subject_uuid:{row.subject_uuid}')
            print(f'session_start_time:{row.session_start_time}')
            print('')

    # Save model parameter data
    df = pd.DataFrame.from_dict(params)
    df.to_csv(PARAMS_FPATH)

    print('Done.')

if __name__ == '__main__':
    main()
