"""Functions for accessing IBL data via DataJoint"""

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import datajoint as dj


class InvalidSessionError(Exception):
    """Exception thrown when session data is invalid."""
    pass


@dataclass
class SessionData:

    block: np.ndarray
    side: np.ndarray
    contrast: np.ndarray
    choice: np.ndarray

    def to_dataframe(self):
        return pd.DataFrame({
            'block': self.block,
            'side': self.side,
            'contrast': self.contrast,
            'choice': self.choice
        })

    @classmethod
    def from_dataframe(cls, df):
        data = df[['block', 'side', 'contrast', 'choice']].to_numpy()
        return cls(data[:, 0], data[:, 1], data[:, 2], data[:, 3])


def get_session_df(uuid: str, timestamp : str) -> pd.DataFrame:
    """Download session dataframe from database and add columns."""
    
    # Need to already be connected to DataJoint database for imports to work
    from ibl_pipeline import behavior, subject
    
    # Select single session
    query_session = (
        behavior.TrialSet
        * subject.Subject
        * subject.SubjectLab
        * subject.SubjectProject 
        & {'subject_uuid': uuid}
        & {'session_start_time': timestamp}
    )
    
    # Select all (conclusive) trials from session
    query_trials = (
        behavior.TrialSet.Trial 
        & query_session
        & 'trial_response_choice !="No Go"'
    )
    
    # Fetch data
    data_dict = query_trials.fetch(
        'trial_id',
        'trial_start_time',
        'trial_end_time',
        'trial_response_time',
        'trial_response_choice', 
        'trial_stim_contrast_left', 
        'trial_stim_contrast_right', 
        'trial_feedback_type',
        'trial_stim_prob_left',
        as_dict=True
    )
    df = pd.DataFrame(data_dict)
    
    # Add columns block, side, contrast, and choice
    df['contrast'] = df['trial_stim_contrast_right'] - df['trial_stim_contrast_left']
    df['choice'] = (df['trial_response_choice'] == "CCW") * 2 - 1
    df['side'] = df['choice'] * df['trial_feedback_type']
    df['block'] = ((0.5 - df['trial_stim_prob_left']) / 0.3).astype('int64')
    
    return df


def validate_session_df(df: pd.DataFrame):

    # Check that all block values are either -1, +1, or 0
    if not df.block.isin([-1, 0, 1]).all():
        raise InvalidSessionError('Trial with invalid block value')

    # Check that first 90 trials all have block = 0
    if not (df.block[:90] == 0).all():
        raise InvalidSessionError('Biased trial in first 90 trials')

    # Check that rest of trials all have block = +-1
    if not df.block[90:].isin([-1, 1]).all():
        raise InvalidSessionError('Unbiased trial after first 90 trials')

    # Check that all side values are either -1 or +1
    if not df.side.isin([-1, 1]).all():
        raise InvalidSessionError('Trial with invalid side value')

    # Check that all choice values are either -1 or +1
    if not df.choice.isin([-1, 1]).all():
        raise InvalidSessionError('Trial with invalid choice value')


def extract_session_data(df: pd.DataFrame):
    """Extract all session data from DataFrame."""

    data = df[['block', 'side', 'contrast', 'choice']].to_numpy()
    return SessionData(
        block=data[:, 0], 
        side=data[:, 1], 
        contrast=data[:, 2], 
        choice=data[:, 3]
    )


def extract_unbiased_data(df: pd.DataFrame):
    """Extract unbiased session data from DataFrame."""

    data = df[['block', 'side', 'contrast', 'choice']].to_numpy()
    return SessionData(
        block=data[0:90, 0], 
        side=data[0:90, 1], 
        contrast=data[0:90, 2], 
        choice=data[0:90, 3]
    )


def get_session(uuid: str, timestamp : str) -> SessionData:
    """Download session and return as SessionData object."""

    # Download session DataFrame and add derived columns
    df = get_session_df(uuid, timestamp)

    # Validate data
    validate_session_df(df)

    # Extract all session data
    sdata = extract_session_data(df)

    return sdata


def get_unbiased_session(uuid: str, timestamp : str) -> SessionData:
    """Download unbiased session data and return as SessionData object."""

    # Download session DataFrame and add derived columns
    df = get_session_df(uuid, timestamp)

    # Validate data
    validate_session_df(df)

    # Extract unbiased block
    sdata = extract_unbiased_data(df)

    return sdata


def save_session_ids_csv(uuids: list[str], timestamps: list[str], fpath: str):
    """Save uuids and timestamps of sessions to CSV file."""

    df = pd.DataFrame({
        'subject_uuid': uuids, 
        'session_start_time': timestamps
    })
    df.to_csv(fpath)


def load_session_ids_csv(fpath: str) -> tuple[list[str], list[str]]:
    """Load uuids and timestamps of sessions from CSV file."""

    # Load dataframe containing session info
    df = pd.read_csv(fpath, index_col=0)

    # Extract ID information from session info
    uuids = []
    timestamps = []
    for row in df.itertuples():
        uuids.append(row.subject_uuid)
        timestamps.append(row.session_start_time)

    return uuids, timestamps


def save_session_csv(sdata: SessionData, fpath: str):
    """Save SessionData object to CSV file."""

    sdata.to_dataframe().to_csv(fpath)


def load_session_csv(fpath: str) -> SessionData:
    """Load SessionData object from CSV file."""

    return SessionData.from_dataframe(pd.read_csv(fpath, index_col=0))


def _timestamp_compress(ts: str) -> str:
    """Convert timestamp from 'YYYY-MM-DD hh:mm:ss' to 'YYYYMMDDThhmmss'"""

    year = ts[0:4]
    month = ts[5:7]
    day = ts[8:10]
    hour = ts[11:13]
    min = ts[14:16]
    sec = ts[17:19]

    return f'{year}{month}{day}T{hour}{min}{sec}'


def _timestamp_expand(dn: str) -> str:
    """Convert timestamp from 'YYYYMMDDThhmmss' to 'YYYY-MM-DD hh:mm:ss'"""

    year = dn[0:4]
    month = dn[4:6]
    day = dn[6:8]
    hour = dn[9:11]
    min = dn[11:13]
    sec = dn[13:15]

    return f'{year}-{month}-{day} {hour}:{min}:{sec}'


def _get_subdir(dirpath: str, uuid: str):
    """Get full path to subdirectory for given subject uuid."""

    return os.path.join(dirpath, uuid)


def _get_fpath(dirpath: str, uuid: str, timestamp: str):
    """Get full path to file for given session."""

    # Subdirectory for subject
    subdir = _get_subdir(dirpath, uuid)

    # Compress timestamp to remove hyphen and colon characters
    fname = _timestamp_compress(timestamp)

    return os.path.join(subdir, fname)


def save_sessions(dirpath: str, uuids: list[str], 
        timestamps: list[str], sdata: list[SessionData]):

    # Create base directory
    os.mkdir(dirpath)

    # Create subdirectories for each subject uuid
    for uuid in set(uuids):
        os.mkdir(_get_subdir(dirpath, uuid))

    # Save all sessions to CSV files
    for i in range(len(uuids)):
        fpath = _get_fpath(dirpath, uuids[i], timestamps[i])
        save_session_csv(sdata[i], fpath)


def get_uuids(dirpath: str) -> list[str]:
    """Get list of unique subject uuids with sessions stored in directory."""

    return os.listdir(dirpath)


def get_timestamps(dirpath: str, uuid: str) -> list[str]:
    """Get list of timestamps for sessions with given uuid."""

    # Get list of CSV filenames in subdirectory for subject uuid
    fnames = os.listdir(_get_subdir(dirpath, uuid))

    # Strip '.csv' suffix from filenames to get compressed timestamp strings
    ts_comp = (os.path.splitext(f)[0] for f in fnames)

    # Add hyphens and colons back to get expanded timestamp strings
    ts_exp = (_timestamp_expand(t) for t in ts_comp)

    return list(ts_exp)


def load_session(dirpath: str, uuid: str, timestamp: str):
    """Load session with uuid and timestamp from directory."""

    return load_session_csv(_get_fpath(dirpath, uuid, timestamp))


def load_subject_sessions(dirpath: str, uuid: str) -> list[SessionData]:
    """Load all sessions for subject with given uuid."""

    timestamps = get_timestamps(dirpath, uuid)
    return [load_session(dirpath, uuid, ts) for ts in timestamps]


def load_all_sessions(dirpath: str) -> list[SessionData]:
    """Load all sessions in directory."""

    all_sessions = []
    for uuid in get_uuids(dirpath):
        all_sessions.extend(load_subject_sessions(dirpath, uuid))

    return all_sessions