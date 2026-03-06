"""Trend modeling and detrending utilities."""

import numpy as np


def fit_linear_trend(df, columns, target):
    """Fit linear trend for target as a function of columns.

    Returns coefficients (including intercept).
    """
    x = np.column_stack([np.ones(len(df))] + [df[col].values for col in columns])
    y = df[target].values
    coeffs, *_ = np.linalg.lstsq(x, y, rcond=None)
    return coeffs


def apply_linear_trend(df, columns, coeffs):
    x = np.column_stack([np.ones(len(df))] + [df[col].values for col in columns])
    return x @ coeffs

