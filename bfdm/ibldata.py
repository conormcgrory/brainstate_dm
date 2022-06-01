"""Functions for accessing IBL data via DataJoint"""

import numpy as np
import pandas as pd
import datajoint as dj


def get_session_data(subject_uuid, session_start_time):
    
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