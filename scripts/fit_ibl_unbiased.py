"""Fit IBL model to all unbiased blocks."""

import numpy as np
import pandas as pd
import datajoint as dj

from bfdm.ibldata import load_session_list_csv, get_unbiased_data
from bfdm.iblmodel import fit_ibl


SESSION_LIST_FPATH = 'data/trained_sessions_test.csv'


def main():

    # Connect to IBL database via DataJoint
    dj.config['database.host'] = 'datajoint-public.internationalbrainlab.org'
    dj.config['database.user'] = 'ibldemo'
    dj.config['database.password'] = 'sfn2019demo'
    dj.conn()

    print('Loading trained sessions...')
    s_list = load_session_list_csv(SESSION_LIST_FPATH)
    print('Done.')

    print('Downloading unbiased session data...')
    sessions = get_unbiased_data(s_list)
    print('Done.')

    print('Fitting IBL model...')
    params = fit_ibl(sessions)
    print('Done.')

    print(params)


if __name__ == '__main__':
    main()
