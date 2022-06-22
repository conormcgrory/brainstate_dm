"""Script for downloading example IBL behavioral data.

The example behavioral dataset from IBL is hosted on a HTTP server,
and can be downloaded as a .zip file. This script (1) downloads the 
data, (2) decompresses it and stores it locally, and (3) creates 
cache files that allow the data to be accessed with the ONE api. 

"""

import io
import zipfile
from pathlib import Path

import requests
from one.api import One


# URL to download IBL behavioral data from
DOWNLOAD_URL = 'https://ndownloader.figshare.com/files/21623715'

# Name of directory stored in .zip file hosted at URL
ZIP_DIRNAME = 'ibl-behavioral-data-Dec2019'

# Local directory where data will be stored
DATA_DIR = 'data/raw/ibl_example'


def main():

    data_dir = Path(DATA_DIR)
    data_root = data_dir.parent

    if not data_root.exists():
        print(f'Error: Desination directory {data_root} does not exist.')
        return -1

    if data_dir.exists():
        print(f'Error: Data directory {data_dir} already exists')
        return -2

    print(f'Downloading compressed data from {DOWNLOAD_URL}...')
    req = requests.get(DOWNLOAD_URL)
    print('Done.')

    print(f'Extracting data to {data_dir}...')
    with zipfile.ZipFile(io.BytesIO(req.content)) as zipped:
        zipped.extractall(data_root)
    Path(data_root, ZIP_DIRNAME).rename(data_dir) 
    print('Done.')

    print('Creating ONE cache files...')
    One.setup(data_dir, hash_files=False)
    print('Done.')


if __name__ == '__main__':
    main()
