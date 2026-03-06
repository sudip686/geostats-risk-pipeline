"""
02_composite.py - Drillhole Compositing

Composites drillhole data to fixed lengths using length-weighted averaging.
Respects lithology boundaries.
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def composite_drills(df, comp_length=1.0, min_comp_length=0.5, grade_field='tgc_pct'):
    """
    Composite drillhole data to fixed lengths.
    Respects hole_id and lith_code boundaries.

    Args:
        df (pd.DataFrame): Dataframe with [hole_id, from_m, to_m, grade, x, y, z, lith_code]
        comp_length (float): Target composite length.
        min_comp_length (float): Minimum length to keep a composite.
        grade_field (str): Name of grade column

    Returns:
        pd.DataFrame: Composited data.
    """
    composites = []

    # Sort by hole and depth
    df = df.sort_values(['hole_id', 'from_m'])

    # Assign unique group for contiguous lithology blocks within each hole
    df['lith_change'] = (
        (df['lith_code'] != df['lith_code'].shift()) |
        (df['hole_id'] != df['hole_id'].shift())
    )
    df['group_id'] = df['lith_change'].cumsum()

    grouped = df.groupby('group_id')

    for _, group in grouped:
        if group.empty:
            continue

        hole_id = group.iloc[0]['hole_id']
        lith_code = group.iloc[0]['lith_code']

        # Total interval for this group
        start_depth = group['from_m'].min()
        end_depth = group['to_m'].max()
        total_len = end_depth - start_depth

        if total_len <= 0:
            continue

        n_comps = int(np.ceil(total_len / comp_length))

        if n_comps == 0:
            continue

        current_from = start_depth

        for i in range(n_comps):
            current_to = min(current_from + comp_length, end_depth)
            actual_len = current_to - current_from

            if actual_len < min_comp_length:
                current_from = current_to
                continue

            # Find assays contributing to this composite
            overlaps = np.maximum(
                0,
                np.minimum(group['to_m'], current_to) - np.maximum(group['from_m'], current_from)
            )
            valid_mask = overlaps > 0.001
            valid_assays = group[valid_mask].copy()
            valid_overlaps = overlaps[valid_mask]

            if valid_assays.empty:
                current_from = current_to
                continue

            # Length-weighted average for grade
            total_weight = valid_overlaps.sum()
            grade = np.sum(valid_assays[grade_field] * valid_overlaps) / total_weight

            # Weighted centroid
            comp_x = np.sum(valid_assays['x'] * valid_overlaps) / total_weight
            comp_y = np.sum(valid_assays['y'] * valid_overlaps) / total_weight
            comp_z = np.sum(valid_assays['z'] * valid_overlaps) / total_weight

            composites.append({
                'hole_id': hole_id,
                'from_m': current_from,
                'to_m': current_to,
                'length': actual_len,
                grade_field: grade,
                'x': comp_x,
                'y': comp_y,
                'z': comp_z,
                'lith_code': lith_code
            })

            current_from = current_to

    result = pd.DataFrame(composites)
    logger.info(f"Compositing complete: {len(result)} composites created")
    return result


def run(data_dir='data', comp_length=2.0, min_comp_length=0.5, grade_field='tgc_pct', output_path=None):
    """
    Run compositing pipeline.

    Args:
        data_dir: Input data directory
        comp_length: Composite length in meters
        min_comp_length: Minimum composite length
        grade_field: Grade column name
        output_path: Optional output CSV path

    Returns:
        pd.DataFrame: Composited data
    """
    from .utils.io import load_data
    from .desurvey import process_holes

    # Load and desurvey data
    collar, survey, assay, litho = load_data(data_dir)
    if collar is None:
        raise ValueError(f"Failed to load data from {data_dir}")

    # Desurvey
    xyz_data = process_holes(collar, survey, assay, litho, grade_field)

    # Composite
    composites = composite_drills(
        xyz_data,
        comp_length=comp_length,
        min_comp_length=min_comp_length,
        grade_field=grade_field
    )

    if output_path:
        composites.to_csv(output_path, index=False)
        logger.info(f"Saved composites to {output_path}")

    return composites
