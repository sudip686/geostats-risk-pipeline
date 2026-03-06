"""
utils/io.py - Input/Output utilities
"""

import os
import json
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def load_data(data_dir):
    """Load standard drillhole CSV files."""
    try:
        collar = pd.read_csv(os.path.join(data_dir, 'collar.csv'))
        survey = pd.read_csv(os.path.join(data_dir, 'survey.csv'))
        assay = pd.read_csv(os.path.join(data_dir, 'assay.csv'))
        litho_path = os.path.join(data_dir, 'lithology.csv')
        if not os.path.exists(litho_path):
            litho_path = os.path.join(data_dir, 'litho.csv')
        litho = pd.read_csv(litho_path)
    except FileNotFoundError as e:
        logger.error(f"Error loading data: {e}")
        return None, None, None, None

    # Standardize columns (strip whitespace, lower case)
    for df in [collar, survey, assay, litho]:
        df.columns = df.columns.str.strip().str.lower()

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


def load_config(config_path):
    """Load YAML configuration."""
    import yaml
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Optional data source switch for repo portability:
    # - local -> paths.local_data_dir
    # - repo/@repo/demo -> paths.repo_data_dir
    data_source = str(cfg.get('data_source', '')).strip().lower().lstrip('@')
    paths_cfg = cfg.get('paths', {})
    if data_source:
        if data_source == 'local':
            cfg['data_dir'] = paths_cfg.get('local_data_dir', cfg.get('data_dir', 'data'))
        elif data_source in {'repo', 'demo'}:
            cfg['data_dir'] = paths_cfg.get('repo_data_dir', 'demo_data')

    # Graceful defaults for portable repo mode when external references are absent.
    if data_source in {'repo', 'demo'}:
        calib = cfg.get('calibration', {})
        ref = calib.get('reference_data')
        if ref and not os.path.exists(ref):
            calib['enabled'] = False
            cfg['calibration'] = calib

        val = cfg.get('validation', {})
        val_ref = val.get('reference_data')
        if val_ref and not os.path.exists(val_ref):
            val.pop('reference_data', None)
            cfg['validation'] = val

        iv = cfg.get('internal_validation', {})
        model_csv = iv.get('model_csv')
        if model_csv and not os.path.exists(model_csv):
            iv['enabled'] = False
            cfg['internal_validation'] = iv

    return cfg


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
