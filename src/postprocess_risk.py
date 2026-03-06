"""
08_postprocess_risk.py - Risk and Uncertainty Postprocessing

Computes:
- P10/P50/P90 percentile grids
- Probability of exceeding cutoff grade
- Risked tonnage curves
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def compute_percentiles(realizations, pvalues=[10, 50, 90]):
    """
    Compute percentile grids from realizations.

    Args:
        realizations (np.array): Shape (n_real, nx, ny, nz)
        pvalues: List of percentile values

    Returns:
        dict: Percentile grids
    """
    result = {}
    for p in pvalues:
        result[f'p{p}'] = np.percentile(realizations, p, axis=0)

    logger.info(f"Computed percentiles: {pvalues}")
    return result


def compute_probability_exceed(realizations, cutoff):
    """
    Compute probability of grade exceeding cutoff.

    Args:
        realizations (np.array): Shape (n_real, nx, ny, nz)
        cutoff: Cutoff grade

    Returns:
        np.array: Probability grid
    """
    prob = np.mean(realizations > cutoff, axis=0)
    logger.info(f"Computed P(grade > {cutoff})")
    return prob


def calculate_tonnage_curve(realizations, cutoffs, volume_per_block, density=2.43, volume_factor=1.0):
    """
    Calculate risked grade-tonnage curves.

    Args:
        realizations (np.array): Shape (n_real, nx, ny, nz)
        cutoffs: Array of cutoff grades
        volume_per_block: Volume of each block (m3)
        density: Rock density (t/m3)

    Returns:
        pd.DataFrame: Tonnage curve data
    """
    results = []

    for cutoff in cutoffs:
        tonnages = []
        grades = []
        contained = []
        nonzero = 0

        for r in realizations:
            mask = r >= cutoff
            n_blocks = np.sum(mask)

            if n_blocks == 0:
                tonnages.append(0)
                grades.append(0)
                contained.append(0)
            else:
                t = n_blocks * volume_per_block * density * volume_factor
                avg_g = np.mean(r[mask])
                c = t * avg_g / 100  # Convert grade % to tonnes

                tonnages.append(t)
                grades.append(avg_g)
                contained.append(c)
                nonzero += 1

        # Sanity checks: mean grade above cutoff and monotonic tonnage
        for avg_g, t in zip(grades, tonnages):
            if t > 0 and avg_g < cutoff:
                raise ValueError(
                    f"Sanity check failed: mean grade {avg_g:.2f}% below cutoff {cutoff:.2f}%"
                )

        results.append({
            'cutoff': cutoff,
            'tonnage_p10': np.percentile(tonnages, 10),
            'tonnage_p50': np.percentile(tonnages, 50),
            'tonnage_p90': np.percentile(tonnages, 90),
            'grade_p10': np.percentile(grades, 10),
            'grade_p50': np.percentile(grades, 50),
            'grade_p90': np.percentile(grades, 90),
            'contained_p10': np.percentile(contained, 10),
            'contained_p50': np.percentile(contained, 50),
            'contained_p90': np.percentile(contained, 90),
            'nonzero_count': nonzero
        })

    # Enforce monotonic tonnage with cutoff
    tonnage_p50 = [row['tonnage_p50'] for row in results]
    if any(t1 < t2 for t1, t2 in zip(tonnage_p50[:-1], tonnage_p50[1:])):
        raise ValueError("Sanity check failed: tonnage increases with cutoff")

    return pd.DataFrame(results)


def run(realizations_path=None, config=None, output_dir='outputs'):
    """
    Run risk postprocessing.

    Args:
        realizations_path: Path to sgs_reals.npy
        config: Configuration dict
        output_dir: Output directory

    Returns:
        dict: Results including percentiles, probability, tonnage
    """
    import os

    # Load realizations
    if realizations_path:
        realizations = np.load(realizations_path)
    else:
        realizations = np.load(os.path.join(output_dir, 'grids', 'sgs_reals.npy'))

    # Optional calibration to block-support reference distribution
    calibration_cfg = config.get('calibration', {}) if config else {}
    if calibration_cfg.get('enabled'):
        from src.calibration import quantile_mapping
        ref_path = calibration_cfg.get('reference_data')
        ref_cols = calibration_cfg.get('reference_columns', {})
        grade_col = ref_cols.get('grade', 'TGC_%')

        if ref_path:
            import pandas as pd
            ref_df = pd.read_csv(ref_path, low_memory=False)
            if grade_col not in ref_df.columns:
                raise ValueError(f"Calibration reference missing column: {grade_col}")
            ref_vals = pd.to_numeric(ref_df[grade_col], errors='coerce').dropna().values
            flat = realizations.reshape(-1)
            mapped = quantile_mapping(flat, ref_vals)
            realizations = mapped.reshape(realizations.shape).astype(realizations.dtype)

            # Save calibrated realizations for traceability
            np.save(os.path.join(output_dir, 'grids', 'sgs_reals_calibrated.npy'), realizations)

    logger.info(f"Loaded realizations: {realizations.shape}")

    # Get parameters
    cutoff = 5.0
    density = 2.43
    volume_factor = 1.0
    if config:
        cutoff = config.get('cutoff_grade', 5.0)
        density = config.get('density_t_per_m3', 2.43)
        volume_factor = config.get('rock_volume_factor', 1.0)

    # Compute volume per block
    # Try to get from meta or config
    grid_meta_path = os.path.join(output_dir, 'grids', 'sgs_meta.json')
    if os.path.exists(grid_meta_path):
        import json
        with open(grid_meta_path, 'r') as f:
            grid_meta = json.load(f)
        dx = grid_meta.get('dx', 25)
        dy = grid_meta.get('dy', 25)
        dz = grid_meta.get('dz', 5)
    else:
        dx, dy, dz = 25, 25, 5

    volume = dx * dy * dz

    # Compute percentiles
    percentiles = compute_percentiles(realizations)

    # Compute probability
    prob = compute_probability_exceed(realizations, cutoff)

    # Calculate tonnage curve
    cutoffs = np.linspace(0, 20, 21)
    tonnage_curve = calculate_tonnage_curve(realizations, cutoffs, volume, density, volume_factor=volume_factor)

    # Save outputs
    grids_dir = os.path.join(output_dir, 'grids')
    tables_dir = os.path.join(output_dir, 'tables')
    os.makedirs(grids_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    np.save(os.path.join(grids_dir, 'p10_grid.npy'), percentiles['p10'])
    np.save(os.path.join(grids_dir, 'p50_grid.npy'), percentiles['p50'])
    np.save(os.path.join(grids_dir, 'p90_grid.npy'), percentiles['p90'])
    np.save(os.path.join(grids_dir, f'prob_gt_{cutoff}.npy'), prob)

    tonnage_curve.to_csv(os.path.join(tables_dir, 'risked_tonnage.csv'), index=False)

    logger.info(f"Saved risk outputs to {output_dir}")

    return {
        'percentiles': percentiles,
        'probability': prob,
        'tonnage_curve': tonnage_curve,
        'cutoff': cutoff
    }
