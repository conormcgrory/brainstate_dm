"""Script for downloading example IBL behavioral data."""


import os
import zipfile

import wget


# URL to download IBL behavioral data from
DOWNLOAD_URL = 'https://ndownloader.figshare.com/files/21623715'

# Name of .zip file containing data
ZIP_FNAME = 'ibl-behavior-data-Dec2019.zip'

# Local directory to store data in (relative to repository root)
DATA_DIR = 'data/raw/ibl_example'


def main():

    # Don't overwrite existing data
    if os.path.exists(DATA_DIR):
        print(f'Error: Data directory {DATA_DIR} already exists')
        return -1

    # Create directory for IBL example data
    os.makedirs(DATA_DIR)
    print(f'Created data directory at {DATA_DIR}.')

    # Download data
    print(f'Downloading data from {DOWNLOAD_URL}...')
    wget.download(DOWNLOAD_URL, DATA_DIR)
    print('\nDone')

    # Extract data from zip file
    zip_fpath = os.path.join(DATA_DIR, ZIP_FNAME)
    print(f'Extracting data from {zip_fpath}...')
    with zipfile.ZipFile(zip_fpath, 'r') as z:
        z.extractall(DATA_DIR)
    print('Done.')


if __name__ == '__main__':
    main()
