"""
utils/io.py - Input/Output utilities
"""

import copy
import os
import json
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)
PROFILE_ENV_VAR = 'TANGA_RUN_PROFILE'


def _standardize_columns(df):
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    return df


def _first_existing_csv(data_dir, preferred_names, contains_tokens, table_name):
    for name in preferred_names:
        path = os.path.join(data_dir, name)
        if os.path.exists(path):
            return path

    candidates = []
    try:
        names = os.listdir(data_dir)
    except OSError as exc:
        logger.error("Error reading data directory %s: %s", data_dir, exc)
        return None

    for name in names:
        lower = name.lower()
        if not lower.endswith('.csv'):
            continue
        if any(token in lower for token in contains_tokens):
            candidates.append(os.path.join(data_dir, name))

    if not candidates:
        logger.error("No %s CSV found in %s", table_name, data_dir)
        return None
    return sorted(candidates)[0]


def _rename_existing(df, mapping):
    rename = {}
    for source, target in mapping.items():
        if source in df.columns and target not in df.columns:
            rename[source] = target
    if rename:
        df = df.rename(columns=rename)
    return df


def _to_numeric(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def _clean_strings(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df.loc[df[col].isin({'', 'nan', 'None'}), col] = np.nan
    return df


def _normalize_collar(collar):
    collar = _standardize_columns(collar)
    collar = _rename_existing(
        collar,
        {
            'bhid': 'hole_id',
            'easting': 'x',
            'northing': 'y',
            'elevation (m)': 'z',
            'elevation': 'z',
            'final_depth (m)': 'total_depth',
            'final_depth': 'total_depth',
        },
    )
    collar = _clean_strings(collar, ['hole_id'])
    return _to_numeric(collar, ['x', 'y', 'z', 'total_depth'])


def _normalize_survey(survey):
    survey = _standardize_columns(survey)
    survey = _rename_existing(
        survey,
        {
            'bhid': 'hole_id',
            'at': 'depth',
            'depth_m': 'depth',
            'final_depth': 'depth',
            'final_depth (m)': 'depth',
            'brg': 'azimuth_deg',
            'azimuth': 'azimuth_deg',
            'dip': 'dip_deg',
            'angle': 'dip_deg',
        },
    )
    survey = _clean_strings(survey, ['hole_id'])
    survey = _to_numeric(survey, ['depth', 'azimuth_deg', 'dip_deg'])
    if 'dip_deg' in survey.columns:
        dips = survey['dip_deg'].dropna()
        if not dips.empty and (dips >= 0).all():
            survey['dip_deg'] = -survey['dip_deg']
    return survey


def _normalize_assay(assay):
    assay = _standardize_columns(assay)
    assay = _rename_existing(
        assay,
        {
            'bhid': 'hole_id',
            'from': 'from_m',
            'to': 'to_m',
            'graphitic carbon': 'tgc_pct',
            'predicted_value': 'tgc_pct',
            'tgc_%': 'tgc_pct',
            'tgc': 'tgc_pct',
            'litho code': 'lith_code',
        },
    )
    assay = _clean_strings(assay, ['hole_id', 'lith_code'])
    return _to_numeric(assay, ['from_m', 'to_m', 'tgc_pct'])


def _normalize_lithology(litho):
    litho = _standardize_columns(litho)
    litho = _rename_existing(
        litho,
        {
            'bhid': 'hole_id',
            'from': 'from_m',
            'to': 'to_m',
            'litho': 'lith_code',
            'lithology': 'lith_code',
        },
    )
    litho = _clean_strings(litho, ['hole_id', 'lith_code', 'weathering'])
    return _to_numeric(litho, ['from_m', 'to_m'])


def load_data(data_dir):
    """Load drillhole CSV files from data_dir and normalize them for the workflow."""
    paths = {
        'collar': _first_existing_csv(data_dir, ['collar.csv', 'OL_collar_p1.csv'], ['collar'], 'collar'),
        'survey': _first_existing_csv(data_dir, ['survey.csv', 'OL_survey_p1.csv'], ['survey'], 'survey'),
        'assay': _first_existing_csv(data_dir, ['assay.csv', 'OL_assay_p1.csv'], ['assay'], 'assay'),
        'lithology': _first_existing_csv(
            data_dir,
            ['lithology.csv', 'litho.csv', 'geology.csv', 'OL_geology_p1.csv'],
            ['lithology', 'litho', 'geology'],
            'lithology',
        ),
    }
    if any(path is None for path in paths.values()):
        return None, None, None, None

    try:
        collar = _normalize_collar(pd.read_csv(paths['collar']))
        survey = _normalize_survey(pd.read_csv(paths['survey']))
        assay = _normalize_assay(pd.read_csv(paths['assay']))
        litho = _normalize_lithology(pd.read_csv(paths['lithology']))
    except (OSError, pd.errors.ParserError) as exc:
        logger.error("Error loading data: %s", exc)
        return None, None, None, None

    logger.info(
        "Loaded data CSVs: collar=%s survey=%s assay=%s lithology=%s",
        os.path.basename(paths['collar']),
        os.path.basename(paths['survey']),
        os.path.basename(paths['assay']),
        os.path.basename(paths['lithology']),
    )
    return collar, survey, assay, litho


def save_grid(grid_dict, output_dir, prefix='sgs'):
    """Save grid to numpy and metadata JSON."""
    os.makedirs(output_dir, exist_ok=True)

    np.save(
        os.path.join(output_dir, f'{prefix}_reals.npy'),
        grid_dict['realizations']
    )

    meta = {
        'x_min': float(grid_dict['x'].min()),
        'x_max': float(grid_dict['x'].max()),
        'y_min': float(grid_dict['y'].min()),
        'y_max': float(grid_dict['y'].max()),
        'z_min': float(grid_dict['z'].min()),
        'z_max': float(grid_dict['z'].max()),
        'dx': float(grid_dict['x'][1] - grid_dict['x'][0]) if len(grid_dict['x']) > 1 else float(grid_dict['dx']),
        'dy': float(grid_dict['y'][1] - grid_dict['y'][0]) if len(grid_dict['y']) > 1 else float(grid_dict['dy']),
        'dz': float(grid_dict['z'][1] - grid_dict['z'][0]) if len(grid_dict['z']) > 1 else float(grid_dict['dz']),
        'shape': grid_dict['realizations'].shape
    }

    with open(os.path.join(output_dir, f'{prefix}_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)


def _deep_merge_dict(base, overrides):
    """Recursively merge dictionaries without mutating the inputs."""
    if not isinstance(overrides, dict):
        return copy.deepcopy(overrides)
    merged = copy.deepcopy(base) if isinstance(base, dict) else {}
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(config_path, profile_name=None):
    """Load YAML configuration and apply an optional runtime profile."""
    import yaml
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f) or {}

    selected_profile = str(profile_name or os.environ.get(PROFILE_ENV_VAR, '') or '').strip()
    if not selected_profile:
        config.setdefault('runtime_profile', None)
        return config

    profiles = config.get('runtime_profiles', {}) or {}
    if selected_profile not in profiles:
        available = ', '.join(sorted(profiles)) or '<none>'
        raise ValueError(f"Unknown runtime profile '{selected_profile}'. Available profiles: {available}")

    profile = profiles[selected_profile] or {}
    overrides = profile.get('overrides', {}) or {}
    merged = _deep_merge_dict(config, overrides)
    merged['runtime_profile'] = {
        'name': selected_profile,
        'description': str(profile.get('description', '') or '').strip(),
    }
    logger.info("Applied runtime profile '%s'", selected_profile)
    return merged


def save_metadata(meta, output_path):
    """Save metadata to JSON."""
    with open(output_path, 'w') as f:
        json.dump(meta, f, indent=2)


def load_npy(path):
    """Load numpy array."""
    return np.load(path)


def save_csv(df, output_path):
    """Save DataFrame to CSV."""
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {output_path}")
