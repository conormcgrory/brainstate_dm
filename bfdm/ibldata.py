"""Functions for handling IBL data."""

import os
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path

import numpy as np
from one.api import One


@dataclass
class RawSessionData:
    """Raw data from 'ibl.fields' table used for each session."""
    
    # EID of session
    eid: str

    # Probability of left stimulus (determines 'bias' of block)
    probability_left: np.ndarray

    # Contrast value if stimulus is on left side, NaN otherwise
    contrast_left: np.ndarray

    # Contrast value if stimulus is on right side, NaN otherwise
    contrast_right: np.ndarray

    # Animal choice (-1 for right, 1 for left)
    choice: np.ndarray

    # Feedback given to animal (-1 for punishment, 1 for reward)
    feedback_type: np.ndarray

    def __eq__(self, other):
        return (self.eid == other.eid 
                and np.array_equal(self.probability_left, other.probability_left)
                and np.array_equal(
                    self.contrast_left, other.contrast_left, equal_nan=True)
                and np.array_equal(
                    self.contrast_right, other.contrast_right, equal_nan=True)
                and np.array_equal(self.choice, other.choice)
                and np.array_equal(self.feedback_type, other.feedback_type))


@dataclass
class SessionData:

    # EID of session
    eid: str

    # Experimental block (-1 for left-biased, 0 for unbiased, +1 for right-biased)
    block: np.ndarray

    # Side that stimulus appears on (-1 for left, +1 for right)
    side: np.ndarray

    # Signed contrast (negative for left stimuli, positive for right)
    contrast: np.ndarray

    # Animal choice (-1 for left, +1 for right)
    choice: np.ndarray

    def __eq__(self, other):
        return (self.eid == other.eid
                and np.array_equal(self.block, other.block)
                and np.array_equal(self.side, other.side)
                and np.array_equal(self.contrast, other.contrast)
                and np.array_equal(self.choice, other.choice))


def get_raw_session_data(eid: str, one: One) -> RawSessionData:
    """Get raw data from ONE database for session with given EID."""

    probability_left = one.load_dataset(eid, '_ibl_trials.probabilityLeft')
    contrast_left = one.load_dataset(eid, '_ibl_trials.contrastLeft')
    contrast_right = one.load_dataset(eid, '_ibl_trials.contrastRight')
    choice = one.load_dataset(eid, '_ibl_trials.choice')
    feedback_type = one.load_dataset(eid, '_ibl_trials.feedbackType')

    return RawSessionData(
        eid=eid,
        probability_left=probability_left,
        contrast_left=contrast_left,
        contrast_right=contrast_right,
        choice=choice,
        feedback_type=feedback_type
    )


def compute_block(probability_left: np.ndarray) -> np.ndarray:
    """Block is -1 for p(left) = 0.8, +1 for p(left) = 0.2."""

    return ((0.5 - probability_left) / 0.3).astype('int64')


def compute_side(choice_raw: np.ndarray, feedback_type: np.ndarray) -> np.ndarray:
    """Side (correct stimulus side) is -1 for left, +1, for right."""

    return -choice_raw * feedback_type


def compute_contrast(c_right: np.ndarray, c_left: np.ndarray) -> np.ndarray:
    """Contrast is difference between right and left stimulus contrast."""

    c_right_num = np.nan_to_num(c_right, nan=0)
    c_left_num = np.nan_to_num(c_left, nan=0)

    return c_right_num - c_left_num


def compute_choice(choice_raw: np.ndarray) -> np.ndarray:
    """Choice is -1 for left, +1 for right (flipped sign of raw choice)."""

    return -choice_raw


def get_processed_data(rdata: RawSessionData) -> SessionData:
    """Extract processed data from entire session."""

    # Compute variables used for model
    block = compute_block(rdata.probability_left)
    side = compute_side(rdata.choice, rdata.feedback_type)
    contrast = compute_contrast(rdata.contrast_right, rdata.contrast_left)
    choice = compute_choice(rdata.choice)

    return SessionData(
        eid=rdata.eid, 
        block=block,
        side=side,
        contrast=contrast,
        choice=choice
    )


def get_processed_unbiased_data(rdata: RawSessionData) -> SessionData:
    """Extract processed data from unbiased block of session."""

    # Compute variables used for model
    block = compute_block(rdata.probability_left)
    side = compute_side(rdata.choice, rdata.feedback_type)
    contrast = compute_contrast(rdata.contrast_right, rdata.contrast_left)
    choice = compute_choice(rdata.choice)

    # Only use trials from unbiased block
    trials_idx = np.where(rdata.probability_left == 0.5)[0]
 
    return SessionData(
        eid=rdata.eid, 
        block=block[trials_idx],
        side=side[trials_idx],
        contrast=contrast[trials_idx],
        choice=choice[trials_idx]
    )


def save_sessions(data: dict[str, list[SessionData]], dirpath: str):
    """Save data from processed sessions to directory."""

    # Create base directory
    os.makedirs(dirpath)

    for subject in data.keys():

        # Create subdirectory for subject
        dpath = os.path.join(dirpath, subject)
        os.mkdir(dpath)

        # Save all sessions to subject directory
        for sdata in data[subject]:
            fname = f'{sdata.eid}.npz'
            fpath = os.path.join(dpath, fname)
            np.savez(
                fpath, 
                block=sdata.block, 
                side=sdata.side, 
                contrast=sdata.contrast, 
                choice=sdata.choice
            )


def load_sessions(dirpath: str) -> dict[str, list[SessionData]]:
    """Load data from processed sessions from directory."""

    data_dpath = Path(dirpath)
    if not data_dpath.is_dir():
        raise ValueError(f'Directory "{dirpath}" not found.')

    data = defaultdict(list)
    for subject_dpath in data_dpath.iterdir():
        for session_fpath in subject_dpath.iterdir():

            # Load session data from .npz file
            sdict = np.load(session_fpath)
            sdata = SessionData(
                eid=session_fpath.stem,
                block=sdict['block'],
                side=sdict['side'],
                contrast=sdict['contrast'],
                choice=sdict['choice']
            )
            
            # Add data to list of sessions for subject
            data[subject_dpath.name].append(sdata)

    return data
