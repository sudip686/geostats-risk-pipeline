"""
Drill spacing sensitivity analysis using existing workflow outputs.
"""

import os
import numpy as np
import pandas as pd
import logging

from src.normal_score import NormalScoreTransform
from src.variography import run as run_variography
from src.sgs import run_sgs
from src.postprocess_risk import calculate_tonnage_curve
from src.utils.io import load_config

logger = logging.getLogger(__name__)


def _subset_holes(df, keep_fraction, seed=42):
    holes = df['hole_id'].unique()
    rng = np.random.default_rng(seed)
    keep = rng.choice(holes, size=max(1, int(len(holes) * keep_fraction)), replace=False)
    return df[df['hole_id'].isin(keep)].copy(), len(holes), len(keep)


def _run_sgs_for_subset(df, config, n_real, output_dir, tag):
    os.makedirs(output_dir, exist_ok=True)
    vario_model, _, _ = run_variography(data_path=None, data_dir=config.get('data_dir', 'data'), config=config)
    nst = NormalScoreTransform()
    weights = df['decluster_weight'].values if 'decluster_weight' in df.columns else None
    nst.fit(df['tgc_pct'].values, weights)
    df = df.copy()
    df['tgc_ns'] = nst.transform(df['tgc_pct'].values)
    grid_def = {
        'x': np.linspace(config['grid']['origin_xyz'][0], config['grid']['origin_xyz'][0] + (config['grid']['nx']-1) * config['grid']['dx'], config['grid']['nx']),
        'y': np.linspace(config['grid']['origin_xyz'][1], config['grid']['origin_xyz'][1] + (config['grid']['ny']-1) * config['grid']['dy'], config['grid']['ny']),
        'z': np.linspace(config['grid']['origin_xyz'][2], config['grid']['origin_xyz'][2] + (config['grid']['nz']-1) * config['grid']['dz'], config['grid']['nz']),
        'dx': config['grid']['dx'],
        'dy': config['grid']['dy'],
        'dz': config['grid']['dz'],
        'nx': config['grid']['nx'],
        'ny': config['grid']['ny'],
        'nz': config['grid']['nz'],
    }
    result = run_sgs(
        df,
        grid_def,
        vario_model,
        nst,
        n_realizations=n_real,
        seed=config.get('simulation', {}).get('seed', 1337),
        n_jobs=config.get('simulation', {}).get('n_jobs', 1),
        chunk_size=config.get('simulation', {}).get('krige_chunk_size', 2000),
        search_radius=None,
        max_neighbors=None,
        min_neighbors=None,
    )
    reals = result['realizations']
    vol = grid_def['dx'] * grid_def['dy'] * grid_def['dz']
    cutoffs = np.linspace(0, 20, 21)
    curve = calculate_tonnage_curve(reals, cutoffs, vol, density=config.get('density_t_per_m3', 2.43))
    curve.to_csv(os.path.join(output_dir, f'risked_tonnage_{tag}.csv'), index=False)
    return curve


def run(config_path='config/project.yaml', output_dir='outputs/sensitivity'):
    config = load_config(config_path)
    sensitivity_cfg = config.get('sensitivity', {})
    if not sensitivity_cfg.get('enabled', True):
        logger.info("Drill spacing sensitivity disabled by config")
        return {}
    domain = pd.read_csv(os.path.join('outputs', 'domain_data.csv'))

    n_real = sensitivity_cfg.get('n_real', 20)
    base_curve = _run_sgs_for_subset(domain, config, n_real=n_real, output_dir=output_dir, tag='base')

    sparse_df, base_holes, sparse_holes = _subset_holes(domain, 0.6, seed=42)
    sparse_curve = _run_sgs_for_subset(sparse_df, config, n_real=n_real, output_dir=output_dir, tag='sparse')

    # Uncertainty width at 5% cutoff
    base_row = base_curve[base_curve['cutoff'] == 5.0].iloc[0]
    sparse_row = sparse_curve[sparse_curve['cutoff'] == 5.0].iloc[0]
    base_width = base_row['tonnage_p90'] - base_row['tonnage_p10']
    sparse_width = sparse_row['tonnage_p90'] - sparse_row['tonnage_p10']
    reduction = (sparse_width - base_width) / sparse_width * 100 if sparse_width > 0 else 0

    report = {
        'base_holes': base_holes,
        'sparse_holes': sparse_holes,
        'base_width_tonnes': base_width,
        'sparse_width_tonnes': sparse_width,
        'uncertainty_reduction_pct': reduction,
    }
    with open(os.path.join(output_dir, 'sensitivity_report.json'), 'w') as f:
        import json
        json.dump(report, f, indent=2)

    logger.info("Saved drill spacing sensitivity report")
    return report


if __name__ == '__main__':
    run()
