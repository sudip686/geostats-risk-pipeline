"""
04_declustering.py - Data Declustering

Applies cell declustering to reduce sampling bias.
Outputs declustered weights and declustered mean.
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def cell_decluster(coords, values, cell_size, min_samples=1):
    """
    Perform cell declustering.

    Args:
        coords: Tuple of (x, y, z) arrays
        values: Array of values to decluster
        cell_size: Tuple of (dx, dy, dz) cell sizes
        min_samples: Minimum samples per cell

    Returns:
        tuple: (declustered_values, weights, cell_counts)
    """
    x, y, z = coords
    dx, dy, dz = cell_size

    # Compute cell indices (use local origin to avoid huge indices)
    ix = np.floor((x - x.min()) / dx).astype(int)
    iy = np.floor((y - y.min()) / dy).astype(int)
    iz = np.floor((z - z.min()) / dz).astype(int)

    # Create unique cell IDs using ravel_multi_index to avoid collisions
    if (ix < 0).any() or (iy < 0).any() or (iz < 0).any():
        raise ValueError("Declustering: negative cell indices detected (check coordinate shift).")

    dims = (int(ix.max()) + 1, int(iy.max()) + 1, int(iz.max()) + 1)
    if any(d <= 0 for d in dims):
        raise ValueError(f"Declustering: invalid grid dims computed: {dims}")

    cell_ids = np.ravel_multi_index((ix, iy, iz), dims=dims)

    # Count samples per cell
    unique_cells, inverse, counts = np.unique(cell_ids, return_inverse=True, return_counts=True)

    # Weight = 1 / count (each sample gets equal weight within cell)
    weights = 1.0 / counts[inverse]

    # Normalize weights to sum to number of samples
    weights = weights * len(values) / weights.sum()

    # Declustered mean
    declustered_mean = np.sum(values * weights) / np.sum(weights)

    return weights, counts[inverse], declustered_mean, len(unique_cells)


def decluster_data(df, cell_size_xy=200, cell_size_z=5, grade_field='tgc_pct'):
    """
    Apply cell declustering to composite data.

    Args:
        df (pd.DataFrame): Composited data with x, y, z, grade
        cell_size_xy: Cell size in X-Y plane (meters)
        cell_size_z: Cell size in Z direction (meters)
        grade_field: Grade column name

    Returns:
        pd.DataFrame: Data with declustering weights added
    """
    coords = (df['x'].values, df['y'].values, df['z'].values)
    values = df[grade_field].values

    # Remove NaNs
    valid = ~np.isnan(values)
    coords = (coords[0][valid], coords[1][valid], coords[2][valid])
    values = values[valid]

    weights, cell_counts, dc_mean, n_cells = cell_decluster(
        coords, values, (cell_size_xy, cell_size_xy, cell_size_z)
    )

    # Add results to dataframe
    result = df.copy()
    result['decluster_weight'] = 1.0  # Default for NaN rows
    result.loc[valid, 'decluster_weight'] = weights
    result['cell_count'] = 1  # Default for NaN rows
    result.loc[valid, 'cell_count'] = cell_counts
    result['declustered_mean'] = dc_mean

    # Calculate declustered statistics
    raw_mean = values.mean()
    logger.info(f"Declustering: raw mean = {raw_mean:.3f}, declustered mean = {dc_mean:.3f}")
    logger.info(f"  Cell size: {cell_size_xy}x{cell_size_xy}x{cell_size_z}m")
    logger.info(f"  Number of occupied cells: {n_cells}")

    return result, {
        'raw_mean': raw_mean,
        'declustered_mean': dc_mean,
        'n_cells': n_cells,
        'cell_size': (cell_size_xy, cell_size_xy, cell_size_z)
    }


def run(data_path=None, data_dir='data', cell_size_xy=200, cell_size_z=5,
        grade_field='tgc_pct', output_path=None):
    """
    Run declustering.

    Args:
        data_path: Path to domain CSV (if None, runs domain filter first)
        data_dir: Input data directory
        cell_size_xy: Declustering cell size X/Y
        cell_size_z: Declustering cell size Z
        grade_field: Grade column name
        output_path: Optional output path

    Returns:
        tuple: (declustered_df, stats_dict)
    """
    if data_path:
        df = pd.read_csv(data_path)
    else:
        from .domains import run as run_domains
        df, _ = run_domains(data_dir=data_dir, grade_field=grade_field)

    result, stats = decluster_data(
        df,
        cell_size_xy=cell_size_xy,
        cell_size_z=cell_size_z,
        grade_field=grade_field
    )

    if output_path:
        result.to_csv(output_path, index=False)
        logger.info(f"Saved declustered data to {output_path}")

    return result, stats
