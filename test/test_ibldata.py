"""Functions for testing bfdm.ibldata module."""

import pytest
import numpy as np
from numpy.testing import assert_array_equal

from bfdm import ibldata


def test_compute_block():

    p_left = np.array([0.2, 0.2, 0.8, 0.5])
    expected = np.array([1, 1, -1, 0])

    result = ibldata.compute_block(p_left)
    assert_array_equal(expected, result)


def test_compute_side():

    choice_raw = np.array([-1, -1, 1, 1])
    feedback_type = np.array([1, -1, -1, 1])
    expected = np.array([1, -1, 1, -1])

    result = ibldata.compute_side(choice_raw, feedback_type)
    assert_array_equal(expected, result)


def test_compute_contrast(): 

    c_right = np.array([0.1, 0.2, np.nan, np.nan])
    c_left = np.array([np.nan, np.nan, 0.5, 0.5])
    expected = np.array([0.1, 0.2, -0.5, -0.5])
    result = ibldata.compute_contrast(c_right, c_left)

    assert_array_equal(expected, result)


def test_compute_choice():

    choice_raw = np.array([-1, 1, -1, 1])
    expected = np.array([1, -1, 1, -1])
    result = ibldata.compute_choice(choice_raw)

    assert_array_equal(expected, result)


def test_get_processed_data():

    rdata = ibldata.RawSessionData(
        eid='001',
        probability_left=np.array([0.5, 0.5, 0.5, 0.2, 0.2]),
        contrast_left=np.array([0.1, 0.1, np.nan, np.nan, 0.5]),
        contrast_right=np.array([np.nan, np.nan, 0.1, 0.1, np.nan]),
        choice=np.array([-1, 1, -1, -1, 1]),
        feedback_type=np.array([-1, 1, 1, 1, 1])
    )
    expected = ibldata.SessionData(
        eid='001',
        block=np.array([0, 0, 0, 1, 1]),
        side=np.array([-1, -1, 1, 1, -1]),
        contrast=np.array([-0.1, -0.1, 0.1, 0.1, -0.5]),
        choice=np.array([1, -1, 1, 1, -1])
    )
    result = ibldata.get_processed_data(rdata)

    assert result == expected


def test_get_processed_unbiased_data():

    rdata = ibldata.RawSessionData(
        eid='001',
        probability_left=np.array([0.5, 0.5, 0.5, 0.2, 0.2]),
        contrast_left=np.array([0.1, 0.1, np.nan, np.nan, 0.5]),
        contrast_right=np.array([np.nan, np.nan, 0.1, 0.1, np.nan]),
        choice=np.array([-1, 1, -1, -1, 1]),
        feedback_type=np.array([-1, 1, 1, 1, 1])
    )
    expected = ibldata.SessionData(
        eid='001',
        block=np.array([0, 0, 0]),
        side=np.array([-1, -1, 1]),
        contrast=np.array([-0.1, -0.1, 0.1]),
        choice=np.array([1, -1, 1])
    )
    result = ibldata.get_processed_unbiased_data(rdata)

    assert result == expected
