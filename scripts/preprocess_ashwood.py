"""Script for selecting and preprocessing data using Ashwood criteria."""

import os

from one.api import ONE


# Local directory where example IBL data is stored 
DATA_DIR = 'data/raw/ibl_example'



def main():

    # Connect ONE database to local data directory
    one = ONE(mode='local', cache_dir=DATA_DIR)

    eids, info = one.search(['_ibl_trials.*'])

    print(eids[0])



if __name__ == '__main__':
    main()
