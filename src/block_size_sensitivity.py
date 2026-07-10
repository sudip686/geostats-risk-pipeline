"""Block size sensitivity for SGS outputs."""

import os
import json
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path
from src.sgs import run_sgs
from src.normal_score import NormalScoreTransform
from src.variography import run as run_variography
from src.postprocess_risk import calculate_tonnage_curve
from src.utils.io import load_config


def run(config_path='config/main_config.yaml', output_dir='outputs/block_size'):
    config = load_config(config_path)
    outputs_root = os.path.dirname(output_dir) if os.path.basename(output_dir) == 'block_size' else output_dir
    data = pd.read_csv(os.path.join(outputs_root, 'domain_data.csv'))
    grade_field = config.get('grade_field', 'tgc_pct')
    if grade_field not in data.columns:
        raise KeyError(f"Missing grade field '{grade_field}' in domain data")

    nst = NormalScoreTransform()
    weights = data['decluster_weight'].values if 'decluster_weight' in data.columns else None
    nst.fit(data[grade_field].values, weights)
    data['tgc_ns'] = nst.transform(data[grade_field].values)

    with tempfile.NamedTemporaryFile(
        mode='w',
        suffix='_block_size_nst.csv',
        prefix='block_size_',
        dir=outputs_root,
        delete=False,
    ) as tf:
        tmp_nst_path = tf.name
    try:
        data.to_csv(tmp_nst_path, index=False)
        vario_model, _, _ = run_variography(
            data_path=tmp_nst_path,
            data_dir=config.get('data_dir', 'data'),
            config=config,
        )
    finally:
        Path(tmp_nst_path).unlink(missing_ok=True)

    base_grid = config['grid']
    grids = {
        'block_100': {'dx': 100, 'dy': 100, 'dz': 10},
        'block_50': {'dx': 50, 'dy': 50, 'dz': 10},
    }

    os.makedirs(output_dir, exist_ok=True)
    results = {}
    for name, dims in grids.items():
        nx = int(base_grid['nx'] * base_grid['dx'] / dims['dx'])
        ny = int(base_grid['ny'] * base_grid['dy'] / dims['dy'])
        nz = int(base_grid['nz'] * base_grid['dz'] / dims['dz'])
        grid_def = {
            'x': np.linspace(base_grid['origin_xyz'][0], base_grid['origin_xyz'][0] + (nx-1)*dims['dx'], nx),
            'y': np.linspace(base_grid['origin_xyz'][1], base_grid['origin_xyz'][1] + (ny-1)*dims['dy'], ny),
            'z': np.linspace(base_grid['origin_xyz'][2], base_grid['origin_xyz'][2] + (nz-1)*dims['dz'], nz),
            'dx': dims['dx'], 'dy': dims['dy'], 'dz': dims['dz'],
            'nx': nx, 'ny': ny, 'nz': nz,
        }

        reals = run_sgs(
            data,
            grid_def,
            vario_model,
            nst,
            n_realizations=10,
            seed=config.get('simulation', {}).get('seed', 1337),
            n_jobs=config.get('simulation', {}).get('n_jobs', 1),
            chunk_size=config.get('simulation', {}).get('krige_chunk_size', 2000),
            search_radius=config.get('simulation', {}).get('search_radius_m'),
            max_neighbors=config.get('simulation', {}).get('max_neighbors'),
            min_neighbors=config.get('simulation', {}).get('min_neighbors'),
            update_every=int(config.get('simulation', {}).get('local_update_every', 1)),
        )['realizations']

        vol = dims['dx'] * dims['dy'] * dims['dz']
        cutoffs = np.linspace(0, 20, 21)
        curve, _ = calculate_tonnage_curve(
            reals, cutoffs, vol, density=config.get('density_t_per_m3', 2.43)
        )
        curve.to_csv(os.path.join(output_dir, f'risked_tonnage_{name}.csv'), index=False)
        results[name] = {
            'dx': dims['dx'], 'dy': dims['dy'], 'dz': dims['dz'],
            'n_real': 10
        }

    with open(os.path.join(output_dir, 'block_size_report.json'), 'w') as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == '__main__':
    run()
