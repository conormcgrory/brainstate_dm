"""Plotting and visualization functions for project."""

import numpy as np

def plot_psychometric(x: np.ndarray, y: np.ndarray):

    x_unique = np.unique(x)
    n_unique = x_unique.shape[0]
    n_pts = x.shape[0]

    prob_y = np.full((n_unique,), np.nan)

    for i in range(n_unique):

        choices =
        prob_y[i] = np.count_nonzero()


