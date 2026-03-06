"""Calibration utilities for block-support alignment."""

import numpy as np


def quantile_mapping(sim_values, ref_values, n_quantiles=1001):
    """Map simulated values to reference distribution via quantile mapping.

    Ensures strictly increasing quantiles to avoid interpolation artifacts.
    """
    sim_values = np.asarray(sim_values)
    ref_values = np.asarray(ref_values)

    probs = np.linspace(0.0, 1.0, n_quantiles)
    sim_q = np.quantile(sim_values, probs)
    ref_q = np.quantile(ref_values, probs)

    # Enforce strict monotonicity
    for i in range(1, len(sim_q)):
        if sim_q[i] <= sim_q[i - 1]:
            sim_q[i] = sim_q[i - 1] + 1e-8
    for i in range(1, len(ref_q)):
        if ref_q[i] <= ref_q[i - 1]:
            ref_q[i] = ref_q[i - 1] + 1e-8

    return np.interp(sim_values, sim_q, ref_q)
