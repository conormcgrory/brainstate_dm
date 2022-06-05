"""Functions for accessing IBL data via DataJoint"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import datajoint as dj


@dataclass
class Session:

    b: np.ndarray
    s: np.ndarray
    x: np.ndarray
    y: np.ndarray

    @classmethod
    def from_dataframe(cls, df):
        data = df[['block', 'correct_side', 'signed_contrast', 'choice']].to_numpy()
        return cls(b=data[:, 0], s=data[:, 1], x=data[:, 2], y=data[:, 3])
    

def get_session_dataframe(subject_uuid, session_start_time):
    
    from ibl_pipeline import behavior, subject
    
    # Select single session
    query_session = (
        behavior.TrialSet
        * subject.Subject
        * subject.SubjectLab
        * subject.SubjectProject 
        & {'subject_uuid': subject_uuid}
        & {'session_start_time': session_start_time}
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
    
    # Add columns for contrast (input) and report right (output)
    df['signed_contrast'] = df['trial_stim_contrast_right'] - df['trial_stim_contrast_left']
    df['choice'] = (df['trial_response_choice'] == "CCW") * 2 - 1
    df['correct_side'] = df['choice'] * df['trial_feedback_type']
    df['block'] = (0.5 - df['trial_stim_prob_left']) / 0.3
    
    return df


def load_session_list_csv(fpath):

    # Load dataframe containing session info
    df = pd.read_csv(fpath, index_col=0)

    # Extract ID information from session info
    sessions = [(row.subject_uuid, row.session_start_time) for row in df.itertuples()]

    return list(sessions)


def get_unbiased_data(session_ids):

    sessions = [] 

    for uuid, start_time in session_ids:

        try:

            # Download data for session
            df = get_session_dataframe(uuid, start_time) 

            # Select rows from unbiased block at beginning of session
            df = df[df.block == 0]

            # If first 90 unbiased trials are all present, add to list
            if len(df) == 90:
                sessions.append(Session.from_dataframe(df))

        except KeyError as e:

            print('')
            print(f'KeyError! Missing required field: {str(e)}')
            print(f'subject_uuid:{uuid}')
            print(f'session_start_time:{start_time}')
            print('')
       
    return sessions
