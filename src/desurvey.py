"""
01_desurvey.py - Drillhole Desurvey

Converts survey data (depth, azimuth, dip) to 3D coordinates
using the Minimum Curvature method.
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def minimum_curvature_desurvey(depths, azimuths, dips, collar_coords):
    """
    Desurvey using Minimum Curvature method.

    Args:
        depths (np.array): Depths of survey points.
        azimuths (np.array): Azimuths in degrees.
        dips (np.array): Dips in degrees (negative = down)
        collar_coords (tuple): (x, y, z) of collar.

    Returns:
        pd.DataFrame: DataFrame with columns [depth, x, y, z]
    """
    idx = np.argsort(depths)
    depths = depths[idx]
    azimuths = np.radians(azimuths[idx])
    dips = np.radians(dips[idx])

    n = len(depths)
    x = np.zeros(n)
    y = np.zeros(n)
    z = np.zeros(n)

    x[0], y[0], z[0] = collar_coords

    for i in range(1, n):
        d_md = depths[i] - depths[i-1]

        az1, dip1 = azimuths[i-1], dips[i-1]
        az2, dip2 = azimuths[i], dips[i]

        # Vector components
        v1 = np.array([
            np.cos(dip1) * np.sin(az1),
            np.cos(dip1) * np.cos(az1),
            np.sin(dip1)
        ])
        v2 = np.array([
            np.cos(dip2) * np.sin(az2),
            np.cos(dip2) * np.cos(az2),
            np.sin(dip2)
        ])

        cos_beta = np.dot(v1, v2)
        cos_beta = np.clip(cos_beta, -1.0, 1.0)
        beta = np.arccos(cos_beta)

        # Minimum curvature factor
        if abs(beta) < 1e-6:
            rf = 1.0
        else:
            rf = 2 / beta * np.tan(beta / 2)

        dx = (d_md / 2) * (v1[0] + v2[0]) * rf
        dy = (d_md / 2) * (v1[1] + v2[1]) * rf
        dz = (d_md / 2) * (v1[2] + v2[2]) * rf

        x[i] = x[i-1] + dx
        y[i] = y[i-1] + dy
        z[i] = z[i-1] + dz

    return pd.DataFrame({'depth': depths, 'x': x, 'y': y, 'z': z})


def interpolate_assays(desurveyed_data, assay_df):
    """
    Calculate XYZ coordinates for assay midpoints.

    Args:
        desurveyed_data (pd.DataFrame): [depth, x, y, z] for the hole.
        assay_df (pd.DataFrame): [from, to, grade]

    Returns:
        pd.DataFrame: Assay DF with x, y, z, length columns.
    """
    assay_df = assay_df.copy()
    assay_df['midpoint'] = (assay_df['from_m'] + assay_df['to_m']) / 2
    assay_df['length'] = assay_df['to_m'] - assay_df['from_m']

    assay_df['x'] = np.interp(
        assay_df['midpoint'],
        desurveyed_data['depth'],
        desurveyed_data['x']
    )
    assay_df['y'] = np.interp(
        assay_df['midpoint'],
        desurveyed_data['depth'],
        desurveyed_data['y']
    )
    assay_df['z'] = np.interp(
        assay_df['midpoint'],
        desurveyed_data['depth'],
        desurveyed_data['z']
    )

    return assay_df


def process_holes(collar, survey, assay, litho, grade_field='tgc_pct'):
    """
    Process all holes: desurvey and merge.

    Args:
        collar: Collar DataFrame
        survey: Survey DataFrame
        assay: Assay DataFrame
        litho: Lithology DataFrame
        grade_field: Name of the grade column

    Returns:
        pd.DataFrame: Combined data with XYZ coordinates
    """
    holes = []
    hole_ids = collar['hole_id'].unique()

    logger.info(f"Processing {len(hole_ids)} holes...")

    for hid in hole_ids:
        c = collar[collar['hole_id'] == hid].iloc[0]
        s = survey[survey['hole_id'] == hid].sort_values('depth')
        a = assay[assay['hole_id'] == hid].sort_values('from_m')
        l = litho[litho['hole_id'] == hid].sort_values('from_m')

        if a.empty:
            logger.warning(f"Skipping hole {hid}: missing assay data")
            continue
        if s.empty:
            total_depth = c.get('total_depth', np.nan)
            if not np.isfinite(total_depth):
                total_depth = a['to_m'].max()
            if not np.isfinite(total_depth) or total_depth <= 0:
                logger.warning(f"Skipping hole {hid}: missing survey data and no usable depth fallback")
                continue
            logger.warning(f"Hole {hid}: missing survey data; using vertical collar-to-depth fallback")
            s = pd.DataFrame({
                'hole_id': [hid],
                'depth': [float(total_depth)],
                'azimuth_deg': [0.0],
                'dip_deg': [-90.0],
            })

        # Add collar as 0 depth survey if missing
        if 0 not in s['depth'].values:
            s0 = pd.DataFrame({
                'hole_id': [hid],
                'depth': [0.0],
                'azimuth_deg': [s.iloc[0]['azimuth_deg']],
                'dip_deg': [s.iloc[0]['dip_deg']]
            })
            s = pd.concat([s0, s], ignore_index=True)

        desurveyed = minimum_curvature_desurvey(
            s['depth'].values,
            s['azimuth_deg'].values,
            s['dip_deg'].values,
            (c['x'], c['y'], c['z'])
        )

        a_xyz = interpolate_assays(desurveyed, a)

        # Assign lithology codes based on midpoint. If the geology table has
        # overlapping intervals, keep the first sorted interval that covers the
        # assay midpoint rather than failing the whole run.
        if 'lith_code' in a_xyz.columns:
            a_xyz['lith_code'] = a_xyz['lith_code'].astype(str).str.strip()
            a_xyz.loc[a_xyz['lith_code'].isin({'', 'nan', 'None'}), 'lith_code'] = 'UNKNOWN'
        else:
            a_xyz['lith_code'] = 'UNKNOWN'

        if not l.empty:
            l = l.sort_values('from_m')
            starts = l['from_m'].to_numpy(dtype=float)
            ends = l['to_m'].to_numpy(dtype=float)
            lith_codes = l['lith_code'].values
            assigned = []
            for midpoint, current_code in zip(a_xyz['midpoint'].to_numpy(dtype=float), a_xyz['lith_code'].values):
                matches = np.flatnonzero((starts <= midpoint) & (midpoint < ends))
                if matches.size:
                    assigned.append(lith_codes[int(matches[0])])
                else:
                    assigned.append(current_code)
            a_xyz['lith_code'] = assigned

        # Ensure grade column is named consistently
        if grade_field != 'tgc_pct':
            if grade_field in a_xyz.columns:
                a_xyz['tgc_pct'] = a_xyz[grade_field]
            else:
                logger.warning(f"Grade field '{grade_field}' not found in assay data")

        holes.append(a_xyz)

    result = pd.concat(holes, ignore_index=True)
    logger.info(f"Desurvey complete: {len(result)} samples processed")

    return result


def run(data_dir='data', output_path=None, grade_field='tgc_pct'):
    """
    Run desurvey on data from directory.

    Args:
        data_dir: Input data directory
        output_path: Optional output CSV path
        grade_field: Name of grade column

    Returns:
        pd.DataFrame: Desurveyed data
    """
    from .utils.io import load_data

    collar, survey, assay, litho = load_data(data_dir)

    if collar is None:
        raise ValueError(f"Failed to load data from {data_dir}")

    result = process_holes(collar, survey, assay, litho, grade_field=grade_field)

    if output_path:
        result.to_csv(output_path, index=False)
        logger.info(f"Saved desurveyed data to {output_path}")

    return result
