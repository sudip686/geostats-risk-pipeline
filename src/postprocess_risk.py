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

from src.support import regularize_realizations

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


def build_classification_benchmark(tonnage_curve: pd.DataFrame, config: dict | None) -> tuple[pd.DataFrame, dict]:
    """Build a simple 15%-at-90%-confidence benchmark summary by cutoff."""
    bench_cfg = (config or {}).get('classification_benchmark', {}) or {}
    enabled = bool(bench_cfg.get('enabled', True))
    rel_limit = float(bench_cfg.get('relative_error_limit', 0.15))
    confidence = float(bench_cfg.get('confidence_interval', 0.90))
    if not enabled:
        return pd.DataFrame(), {
            'enabled': False,
            'relative_error_limit': rel_limit,
            'confidence_interval': confidence,
        }

    def _rel_half_width(p_lo, p_mid, p_hi):
        denom = max(abs(float(p_mid)), 1e-9)
        return max(abs(float(p_hi) - float(p_mid)), abs(float(p_mid) - float(p_lo))) / denom

    rows = []
    for _, row in tonnage_curve.iterrows():
        tonnage_rel = _rel_half_width(row['tonnage_p05'], row['tonnage_p50'], row['tonnage_p95'])
        grade_rel = _rel_half_width(row['grade_p05'], row['grade_p50'], row['grade_p95'])
        contained_rel = _rel_half_width(row['contained_p05'], row['contained_p50'], row['contained_p95'])
        rows.append({
            'cutoff': float(row['cutoff']),
            'tonnage_rel_half_width_90': float(tonnage_rel),
            'grade_rel_half_width_90': float(grade_rel),
            'contained_rel_half_width_90': float(contained_rel),
            'tonnage_within_15pct_90ci': bool(tonnage_rel <= rel_limit),
            'grade_within_15pct_90ci': bool(grade_rel <= rel_limit),
            'contained_within_15pct_90ci': bool(contained_rel <= rel_limit),
            'measured_candidate_proxy': bool(
                tonnage_rel <= rel_limit and grade_rel <= rel_limit and contained_rel <= rel_limit
            ),
        })

    df = pd.DataFrame(rows)
    target_cutoff = float(bench_cfg.get('cutoff_grade', (config or {}).get('cutoff_grade', 3.0)))
    nearest_idx = (df['cutoff'] - target_cutoff).abs().idxmin()
    summary = {
        'enabled': True,
        'relative_error_limit': rel_limit,
        'confidence_interval': confidence,
        'target_cutoff_grade': target_cutoff,
        'selected_cutoff_grade': float(df.loc[nearest_idx, 'cutoff']),
        'selected_row': {
            k: (bool(v) if isinstance(v, (bool, np.bool_)) else float(v))
            for k, v in df.loc[nearest_idx].to_dict().items()
        },
        'interpretation': (
            "This is a quantitative screening proxy for the 15%-at-90%-confidence benchmark; "
            "it is not a formal reporting-code classification by itself."
        ),
    }
    return df, summary


def compute_probability_exceed(realizations, cutoff):
    """
    Compute probability of grade exceeding cutoff.

    Args:
        realizations (np.array): Shape (n_real, nx, ny, nz)
        cutoff: Cutoff grade

    Returns:
        np.array: Probability grid
    """
    valid = np.isfinite(realizations)
    exceed = valid & (realizations > cutoff)
    prob = np.sum(exceed, axis=0) / np.maximum(np.sum(valid, axis=0), 1)
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
        volume_factor: Deprecated. Ignored; tonnage uses full block volume x density.

    Returns:
        pd.DataFrame: Tonnage curve data
    """
    results = []
    realization_rows = []
    if volume_factor not in (None, 1.0):
        logger.warning(
            "Ignoring deprecated rock volume factor %.6g; tonnage is computed as block count x block volume x density.",
            float(volume_factor),
        )
    volume_factor = 1.0

    for cutoff in cutoffs:
        tonnages = []
        grades = []
        contained = []
        nonzero = 0

        for ridx, r in enumerate(realizations):
            valid = np.isfinite(r)
            mask = valid & (r >= cutoff)
            n_blocks = np.sum(mask)

            if n_blocks == 0:
                t = 0.0
                avg_g = 0.0
                c = 0.0
                tonnages.append(t)
                grades.append(avg_g)
                contained.append(c)
            else:
                t = n_blocks * volume_per_block * density
                avg_g = np.mean(r[mask])
                c = t * avg_g / 100  # Convert grade % to tonnes

                tonnages.append(t)
                grades.append(avg_g)
                contained.append(c)
                nonzero += 1
            realization_rows.append(
                {
                    'cutoff': float(cutoff),
                    'realization_id': int(ridx),
                    'tonnage': float(t),
                    'avg_grade': float(avg_g),
                    'contained': float(c),
                }
            )

        # Sanity checks: mean grade above cutoff and monotonic tonnage
        for avg_g, t in zip(grades, tonnages):
            if t > 0 and avg_g < cutoff:
                raise ValueError(
                    f"Sanity check failed: mean grade {avg_g:.2f}% below cutoff {cutoff:.2f}%"
                )

        results.append({
            'cutoff': cutoff,
            'tonnage_p05': np.percentile(tonnages, 5),
            'tonnage_mean': float(np.mean(tonnages)),
            'tonnage_p10': np.percentile(tonnages, 10),
            'tonnage_p50': np.percentile(tonnages, 50),
            'tonnage_p90': np.percentile(tonnages, 90),
            'tonnage_p95': np.percentile(tonnages, 95),
            'grade_p05': np.percentile(grades, 5),
            'grade_mean': float(np.mean(grades)),
            'grade_p10': np.percentile(grades, 10),
            'grade_p50': np.percentile(grades, 50),
            'grade_p90': np.percentile(grades, 90),
            'grade_p95': np.percentile(grades, 95),
            'contained_p05': np.percentile(contained, 5),
            'contained_mean': float(np.mean(contained)),
            'contained_p10': np.percentile(contained, 10),
            'contained_p50': np.percentile(contained, 50),
            'contained_p90': np.percentile(contained, 90),
            'contained_p95': np.percentile(contained, 95),
            'nonzero_count': nonzero
        })

    # Enforce monotonic tonnage with cutoff
    tonnage_p50 = [row['tonnage_p50'] for row in results]
    if any(t1 < t2 for t1, t2 in zip(tonnage_p50[:-1], tonnage_p50[1:])):
        raise ValueError("Sanity check failed: tonnage increases with cutoff")

    return pd.DataFrame(results), pd.DataFrame(realization_rows)


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

    reporting_grid_def = None
    reporting_grid_meta = None

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
        legacy_volume_factor = float(config.get('rock_volume_factor', 1.0))
        if legacy_volume_factor != 1.0:
            logger.warning(
                "Config rock_volume_factor=%.6g is deprecated and will be ignored; "
                "tonnage uses full block volume x density.",
                legacy_volume_factor,
            )
            config['deprecated_rock_volume_factor'] = legacy_volume_factor
            config['rock_volume_factor'] = 1.0

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
        grid_meta = {
            'x_min': 0.0,
            'y_min': 0.0,
            'z_min': 0.0,
            'dx': dx,
            'dy': dy,
            'dz': dz,
            'nx': realizations.shape[1],
            'ny': realizations.shape[2],
            'nz': realizations.shape[3],
        }

    sim_grid_def = {
        'x': grid_meta['x_min'] + np.arange(int(grid_meta['nx'])) * float(grid_meta['dx']),
        'y': grid_meta['y_min'] + np.arange(int(grid_meta['ny'])) * float(grid_meta['dy']),
        'z': grid_meta['z_min'] + np.arange(int(grid_meta['nz'])) * float(grid_meta['dz']),
        'dx': float(grid_meta['dx']),
        'dy': float(grid_meta['dy']),
        'dz': float(grid_meta['dz']),
        'nx': int(grid_meta['nx']),
        'ny': int(grid_meta['ny']),
        'nz': int(grid_meta['nz']),
    }

    regularized, reporting_grid_def, reporting_grid_meta = regularize_realizations(realizations, sim_grid_def, config)
    if regularized is not None:
        logger.info(
            "Regularized realizations from %.1fx%.1fx%.1f m to %.1fx%.1fx%.1f m support",
            sim_grid_def['dx'],
            sim_grid_def['dy'],
            sim_grid_def['dz'],
            reporting_grid_def['dx'],
            reporting_grid_def['dy'],
            reporting_grid_def['dz'],
        )
        realizations = regularized
        np.save(os.path.join(output_dir, 'grids', 'sgs_reals_reporting.npy'), realizations)
        import json
        with open(os.path.join(output_dir, 'grids', 'sgs_reporting_meta.json'), 'w') as f:
            json.dump(reporting_grid_meta, f, indent=2)
        dx = reporting_grid_def['dx']
        dy = reporting_grid_def['dy']
        dz = reporting_grid_def['dz']

    volume = dx * dy * dz

    # Compute percentiles
    percentiles = compute_percentiles(realizations, pvalues=[5, 10, 50, 90, 95])

    # Compute probability
    prob = compute_probability_exceed(realizations, cutoff)

    # Calculate tonnage curve
    risk_cfg = config.get('risk', {}) if config else {}
    if risk_cfg.get('cutoffs'):
        cutoffs = np.array([float(v) for v in risk_cfg['cutoffs']], dtype=float)
    else:
        cutoffs = np.linspace(0, 20, 21)
    tonnage_curve, tonnage_by_realization = calculate_tonnage_curve(
        realizations, cutoffs, volume, density, volume_factor=volume_factor
    )
    class_benchmark, class_summary = build_classification_benchmark(tonnage_curve, config)

    # Save outputs
    grids_dir = os.path.join(output_dir, 'grids')
    tables_dir = os.path.join(output_dir, 'tables')
    os.makedirs(grids_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    np.save(os.path.join(grids_dir, 'p10_grid.npy'), percentiles['p10'])
    np.save(os.path.join(grids_dir, 'p50_grid.npy'), percentiles['p50'])
    np.save(os.path.join(grids_dir, 'p90_grid.npy'), percentiles['p90'])
    np.save(os.path.join(grids_dir, 'p05_grid.npy'), percentiles['p5'])
    np.save(os.path.join(grids_dir, 'p95_grid.npy'), percentiles['p95'])
    np.save(os.path.join(grids_dir, f'prob_gt_{cutoff}.npy'), prob)

    tonnage_curve.to_csv(os.path.join(tables_dir, 'risked_tonnage.csv'), index=False)
    tonnage_by_realization.to_csv(os.path.join(tables_dir, 'risked_tonnage_by_realization.csv'), index=False)
    if not class_benchmark.empty:
        class_benchmark.to_csv(os.path.join(tables_dir, 'classification_benchmark.csv'), index=False)
        import json
        with open(os.path.join(tables_dir, 'classification_benchmark_summary.json'), 'w', encoding='utf-8') as f:
            json.dump(class_summary, f, indent=2)

    logger.info(f"Saved risk outputs to {output_dir}")

    return {
        'percentiles': percentiles,
        'probability': prob,
        'tonnage_curve': tonnage_curve,
        'classification_benchmark': class_benchmark,
        'classification_summary': class_summary,
        'cutoff': cutoff
    }
