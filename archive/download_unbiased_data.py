"""Script for downloading unbiased blocks of session data."""

import datajoint as dj

from bfdm import ibldata


# List of uuids and timestamps of 'trained' sessions
TRAINED_LIST_FPATH = 'data/trained_sessions.csv'

# Directory where session data is stored
DATA_DIRPATH = 'data/unbiased'


def main():

    # Connect to IBL database via DataJoint
    dj.config['database.host'] = 'datajoint-public.internationalbrainlab.org'
    dj.config['database.user'] = 'ibldemo'
    dj.config['database.password'] = 'sfn2019demo'
    dj.conn()

    print(f'Loading session IDs from {TRAINED_LIST_FPATH}...')
    uuids, timestamps = ibldata.load_session_ids_csv(TRAINED_LIST_FPATH)
    print('Done.')

    print('Downloading and validating session data...')
    sessions_save = [] 
    uuids_save = []
    timestamps_save = []
    for uuid, ts in zip(uuids, timestamps):
        try:
            sdata = ibldata.get_unbiased_session(uuid, ts) 
            sessions_save.append(sdata)
            uuids_save.append(uuid)
            timestamps_save.append(ts)
        except ibldata.InvalidSessionError as e:
            print(f"Invalid session: uuid='{uuid}', ts='{ts}'")
            print(f"Error:'{str(e)}'")
        except KeyError as e:
            print(f"KeyError: uuid='{uuid}', ts='{ts}'")
            print(f"Error:'{str(e)}'")
    print('Done.')

    print(f'Saving data to {DATA_DIRPATH}...')
    ibldata.save_sessions(DATA_DIRPATH, uuids_save, timestamps_save, sessions_save)
    print('Done.')

    save_id_fpath = 'trained_sessions_clean.csv'
    print('Saving session IDs...')
    ibldata.save_session_ids_csv(uuids_save, timestamps_save, save_id_fpath)
    print('Done.')


if __name__ == '__main__':
    main()
