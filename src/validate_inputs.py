"""
00_validate_inputs.py - Input Data QA/QC

Validates drillhole data for:
- Missing required columns
- Overlapping assay intervals
- Assay intervals beyond hole depth
- Non-monotonic survey depths
- Invalid azimuth/dip ranges
- Missing lithology codes
"""

import numpy as np
import pandas as pd
import logging
import os

logger = logging.getLogger(__name__)


def validate_collar(collar_df):
    """Validate collar data."""
    issues = []
    required_cols = ['hole_id', 'x', 'y', 'z']

    for col in required_cols:
        if col not in collar_df.columns:
            issues.append(f"Missing required column: {col}")

    if collar_df.empty:
        issues.append("Collar data is empty")
        return issues

    # Check for duplicate hole IDs
    dupes = collar_df['hole_id'].duplicated().sum()
    if dupes > 0:
        issues.append(f"Found {dupes} duplicate hole IDs")

    # Check for null coordinates
    null_coords = collar_df[['x', 'y', 'z']].isnull().any(axis=1).sum()
    if null_coords > 0:
        issues.append(f"Found {null_coords} rows with null coordinates")

    return issues


def validate_survey(survey_df):
    """Validate survey data."""
    issues = []
    required_cols = ['hole_id', 'depth', 'azimuth_deg', 'dip_deg']

    for col in required_cols:
        if col not in survey_df.columns:
            issues.append(f"Missing required column: {col}")

    if survey_df.empty:
        issues.append("Survey data is empty")
        return issues

    # Check for non-monotonic depths per hole
    for hole_id, group in survey_df.groupby('hole_id'):
        depths = group.sort_values('depth')['depth'].values
        if not np.all(np.diff(depths) > 0):
            issues.append(f"Hole {hole_id}: Non-monotonic survey depths")

    # Check azimuth range
    invalid_az = ((survey_df['azimuth_deg'] < 0) | (survey_df['azimuth_deg'] > 360)).sum()
    if invalid_az > 0:
        issues.append(f"Found {invalid_az} rows with invalid azimuth (should be 0-360)")

    # Check dip range
    invalid_dip = ((survey_df['dip_deg'] < -90) | (survey_df['dip_deg'] > 90)).sum()
    if invalid_dip > 0:
        issues.append(f"Found {invalid_dip} rows with invalid dip (should be -90 to 90)")

    return issues


def validate_assay(assay_df, collar_df):
    """Validate assay data."""
    issues = []
    required_cols = ['hole_id', 'from_m', 'to_m']

    for col in required_cols:
        if col not in assay_df.columns:
            issues.append(f"Missing required column: {col}")

    if assay_df.empty:
        issues.append("Assay data is empty")
        return issues

    # Check for negative intervals
    neg_intervals = (assay_df['to_m'] <= assay_df['from_m']).sum()
    if neg_intervals > 0:
        issues.append(f"Found {neg_intervals} rows with non-positive interval length")

    # Check for overlapping intervals per hole
    for hole_id, group in assay_df.groupby('hole_id'):
        sorted_group = group.sort_values('from_m')
        froms = sorted_group['from_m'].values
        tos = sorted_group['to_m'].values
        for i in range(1, len(froms)):
            if froms[i] < tos[i-1]:
                issues.append(f"Hole {hole_id}: Overlapping assay intervals detected")
                break

    # Check intervals beyond hole depth
    if not collar_df.empty:
        max_depths = collar_df.groupby('hole_id')['total_depth'].max() if 'total_depth' in collar_df.columns else None
        if max_depths is not None:
            assay_with_depth = assay_df.merge(
                max_depths.reset_index().rename(columns={'total_depth': 'max_depth'}),
                on='hole_id', how='left'
            )
            beyond_depth = (assay_with_depth['to_m'] > assay_with_depth['max_depth']).sum()
            if beyond_depth > 0:
                issues.append(f"Found {beyond_depth} intervals beyond hole total depth")

    return issues


def validate_lithology(litho_df):
    """Validate lithology data."""
    issues = []
    required_cols = ['hole_id', 'from_m', 'to_m', 'lith_code']

    for col in required_cols:
        if col not in litho_df.columns:
            issues.append(f"Missing required column: {col}")

    if litho_df.empty:
        issues.append("Lithology data is empty")
        return issues

    # Check for negative intervals
    neg_intervals = (litho_df['to_m'] <= litho_df['from_m']).sum()
    if neg_intervals > 0:
        issues.append(f"Found {neg_intervals} lithology rows with non-positive interval length")

    return issues


def run_validation(data_dir='data'):
    """
    Run full validation pipeline.

    Returns:
        dict: Validation report with counts and issues
    """
    logger.info("Starting input validation...")

    report = {
        'passed': True,
        'n_holes': 0,
        'n_surveys': 0,
        'n_assays': 0,
        'n_lithologies': 0,
        'total_meters': 0,
        'issues': []
    }

    try:
        collar = pd.read_csv(f"{data_dir}/collar.csv")
        survey = pd.read_csv(f"{data_dir}/survey.csv")
        assay = pd.read_csv(f"{data_dir}/assay.csv")
        litho = pd.read_csv(f"{data_dir}/lithology.csv")
    except FileNotFoundError as e:
        report['issues'].append(f"Data file not found: {e}")
        report['passed'] = False
        return report

    # Standardize column names
    for df in [collar, survey, assay, litho]:
        df.columns = df.columns.str.strip().str.lower()

    # Validate each table
    report['issues'].extend(validate_collar(collar))
    report['issues'].extend(validate_survey(survey))
    report['issues'].extend(validate_assay(assay, collar))
    report['issues'].extend(validate_lithology(litho))

    # Calculate statistics
    report['n_holes'] = collar['hole_id'].nunique()
    report['n_surveys'] = len(survey)
    report['n_assays'] = len(assay)
    report['n_lithologies'] = len(litho)

    if not assay.empty:
        assay['length'] = assay['to_m'] - assay['from_m']
        report['total_meters'] = assay['length'].sum()

    if report['issues']:
        report['passed'] = False

    # Grid overlap check (if config exists)
    try:
        from src.utils.io import load_config
        config = load_config('config/project.yaml')
        grid = config.get('grid', {})
        origin = grid.get('origin_xyz')
        nx, ny, nz = grid.get('nx'), grid.get('ny'), grid.get('nz')
        dx, dy, dz = grid.get('dx'), grid.get('dy'), grid.get('dz')
        if origin and nx and ny and nz and dx and dy and dz:
            x0, y0, z0 = origin
            x1 = x0 + (nx - 1) * dx
            y1 = y0 + (ny - 1) * dy
            z1 = z0 + (nz - 1) * dz

            data_x = collar['x'].min(), collar['x'].max()
            data_y = collar['y'].min(), collar['y'].max()
            data_z = collar['z'].min(), collar['z'].max()

            overlap_x = max(0.0, min(x1, data_x[1]) - max(x0, data_x[0]))
            overlap_y = max(0.0, min(y1, data_y[1]) - max(y0, data_y[0]))
            overlap_z = max(0.0, min(z1, data_z[1]) - max(z0, data_z[0]))

            data_extent_x = max(1e-6, data_x[1] - data_x[0])
            data_extent_y = max(1e-6, data_y[1] - data_y[0])
            data_extent_z = max(1e-6, data_z[1] - data_z[0])

            frac_x = overlap_x / data_extent_x
            frac_y = overlap_y / data_extent_y
            frac_z = overlap_z / data_extent_z

            if min(frac_x, frac_y, frac_z) < 0.05:
                report['issues'].append(
                    "Grid/data overlap < 5% in at least one axis. "
                    "Check grid origin/size vs data coordinates."
                )
                report['passed'] = False

            # Log sample coverage within grid
            in_grid = (
                (collar['x'] >= x0) & (collar['x'] <= x1) &
                (collar['y'] >= y0) & (collar['y'] <= y1) &
                (collar['z'] >= z0) & (collar['z'] <= z1)
            )
            coverage = float(in_grid.sum() / max(1, len(collar))) * 100
            logger.info(f"Collar samples inside grid: {coverage:.1f}%")

            # Domain sample coverage (if domain data exists)
            domain_path = os.path.join('outputs', 'domain_data.csv')
            if os.path.exists(domain_path):
                domain = pd.read_csv(domain_path)
                domain_in = (
                    (domain['x'] >= x0) & (domain['x'] <= x1) &
                    (domain['y'] >= y0) & (domain['y'] <= y1) &
                    (domain['z'] >= z0) & (domain['z'] <= z1)
                )
                d_cov = float(domain_in.sum() / max(1, len(domain))) * 100
                logger.info(f"Domain samples inside grid: {d_cov:.1f}%")
    except Exception as exc:
        logger.warning(f"Grid overlap check skipped: {exc}")

    # Log summary
    logger.info(f"Validation complete: {report['n_holes']} holes, {report['total_meters']:.1f} meters")
    if report['issues']:
        for issue in report['issues']:
            logger.warning(f"  - {issue}")
    else:
        logger.info("  All checks passed!")

    return report


if __name__ == '__main__':
    import sys
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'data'
    report = run_validation(data_dir)
    sys.exit(0 if report['passed'] else 1)
