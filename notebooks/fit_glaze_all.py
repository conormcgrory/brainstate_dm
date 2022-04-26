"""Fit Glaze model to all sessions."""

from itertools import islice

import numpy as np
import pandas as pd
import datajoint as dj

from ibldata import get_session_data
from glazemodel import GlazeModel


SESSION_LIST_FPATH = '../data/ibl/trained_sessions.csv'
PARAMS_FPATH = '../data/ibl/glaze_params.csv'
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
    params = {'h': [], 'w_0': [], 'w_1': []}

    for row in sessions_df.itertuples():

        try:

            # Download data for session
            df = get_session_data(row.subject_uuid, row.session_start_time) 
            data = df[['signed_contrast', 'choice']].to_numpy()
            x = data[:, 0]
            y = data[:, 1]

            # Fit Glaze model
            model = GlazeModel()
            model.fit(x, y)

            # Add params to dict
            params['h'].append(model.h)
            params['w_0'].append(model.w_0)
            params['w_1'].append(model.w_1)
    
            # Print results
            print(f'subject_uuid:{row.subject_uuid}')
            print(f'session_start_time:{row.session_start_time}')
            print(f'h: {model.h}')
            print(f'w_0: {model.w_0}')
            print(f'w_1: {model.w_1}') 
            print('')

        except KeyError:

            print('')
            print('KeyError! Missing required field!')
            print(f'subject_uuid:{row.subject_uuid}')
            print(f'session_start_time:{row.session_start_time}')
            print('')

    # Save model parameter data
    df = pd.DataFrame.from_dict(params)
    df.to_csv(PARAMS_FPATH)

    print('Done.')

if __name__ == '__main__':
    main()
