"""Fit IBL model to all unbiased blocks."""

from bfdm import ibldata, iblmodel


DATA_DIRPATH = 'data/unbiased_test'


def main():

    print(f'Loading session data from {DATA_DIRPATH}...')
    sessions = ibldata.load_all_sessions(DATA_DIRPATH)
    print('Done.')

    print('Fitting IBL model...')
    s = [a.side for a in sessions]
    x = [a.contrast for a in sessions]
    y = [a.choice for a in sessions]
    params = iblmodel.fit_ibl(s, x, y)
    print('Done.')

    print(params)


if __name__ == '__main__':
    main()
