"""
Categorical domain simulation for geology-led stochastic domaining.

This module creates fine-grid realizations of categorical geology domains
from drillhole composites using a lightweight sequential categorical
simulation scheme with fixed orebody-oriented neighborhoods.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .domains import build_categorical_domain_data, canonical_domain_groups
from .sgs import define_grid
from .variography import build_orebody_axes, orebody_from_config

logger = logging.getLogger(__name__)


def _domain_categories(config=None) -> list[str]:
    groups = canonical_domain_groups(config=config)
    ordered = [name for name in groups.keys() if name != 'host_waste']
    if 'host_waste' in groups:
        ordered.append('host_waste')
    elif 'host_waste' not in ordered:
        ordered.append('host_waste')
    return ordered


def _project_to_orebody_axes(coords: np.ndarray, config: dict | None) -> np.ndarray:
    orebody = orebody_from_config(config)
    if not orebody:
        return coords.astype(float)

    strike_deg = orebody.get('strike_deg')
    dip_deg = orebody.get('dip_deg')
    dip_direction_deg = orebody.get('dip_direction_deg')
    dip_positive_down = bool(orebody.get('dip_positive_down', True))
    if strike_deg is None or dip_deg is None:
        return coords.astype(float)

    axes = build_orebody_axes(
        float(strike_deg),
        float(dip_deg),
        float(dip_direction_deg) if dip_direction_deg is not None else None,
        dip_positive_down=dip_positive_down,
    )
    basis = np.vstack([axes['strike'], axes['dip'], axes['normal']])
    return coords @ basis.T


def _categorical_radii(config=None) -> np.ndarray:
    domains_cfg = (config or {}).get('domains', {}) or {}
    radii = domains_cfg.get('search_radius_m')
    if radii is None:
        radii = (config or {}).get('simulation', {}).get('search_radius_m', [250, 200, 20])
    arr = np.asarray(radii, dtype=float).reshape(-1)
    if arr.size == 1:
        arr = np.repeat(arr, 3)
    if arr.size != 3 or (arr <= 0).any():
        raise ValueError(f"Invalid categorical domain search radii: {radii}")
    return arr


def _search_points(coords: np.ndarray, config=None) -> tuple[np.ndarray, float]:
    radii = _categorical_radii(config=config)
    projected = _project_to_orebody_axes(coords, config=config)
    scaled = projected / radii.reshape(1, 3)
    return scaled.astype(float), 1.0


def _local_probabilities(
    cond_search_points: np.ndarray,
    cond_ids: np.ndarray,
    grid_search_points: np.ndarray,
    search_radius: float,
    max_neighbors: int,
    priors: np.ndarray,
    prior_weight: float,
    n_categories: int,
    host_category_idx: int | None = None,
) -> np.ndarray:
    """Compute local categorical probabilities at each grid node."""
    tree = cKDTree(cond_search_points)
    n_cond = cond_search_points.shape[0]
    dists, idx = tree.query(
        grid_search_points,
        k=max_neighbors,
        distance_upper_bound=search_radius,
        workers=-1,
    )
    dists = np.asarray(dists, dtype=float)
    idx = np.asarray(idx, dtype=int)
    if dists.ndim == 1:
        dists = dists[:, None]
        idx = idx[:, None]

    n_nodes = grid_search_points.shape[0]
    scores = np.zeros((n_nodes, n_categories), dtype=np.float32)
    support_counts = np.zeros((n_nodes, n_categories), dtype=np.int16)
    nearest_dist = np.full(n_nodes, np.inf, dtype=np.float32)
    for col in range(idx.shape[1]):
        valid = idx[:, col] < n_cond
        if not np.any(valid):
            continue
        valid_rows = np.flatnonzero(valid)
        valid_dists = dists[valid, col]
        valid_ids = cond_ids[idx[valid, col]]
        nearest_dist[valid_rows] = np.minimum(nearest_dist[valid_rows], valid_dists.astype(np.float32))
        weights = 1.0 / np.maximum(valid_dists, 1e-6)
        np.add.at(scores, (valid_rows, valid_ids), weights.astype(np.float32))
        np.add.at(support_counts, (valid_rows, valid_ids), 1)

    has_local_support = support_counts.sum(axis=1) > 0
    if float(prior_weight) > 0.0:
        supported_categories = support_counts > 0
        scores += np.where(supported_categories, (priors.astype(np.float32) * float(prior_weight))[None, :], 0.0)

    if host_category_idx is not None:
        bounded = np.clip(nearest_dist, 0.0, float(search_radius))
        host_scores = (bounded / max(float(search_radius), 1e-6)) * float(prior_weight)
        scores[has_local_support, host_category_idx] += host_scores[has_local_support].astype(np.float32)

    no_support = ~has_local_support
    if np.any(no_support):
        scores[no_support] = 0.0
        if host_category_idx is not None:
            scores[no_support, host_category_idx] = 1.0
        else:
            scores[no_support] = 1.0

    zero = scores.sum(axis=1) <= 0
    if np.any(zero):
        scores[zero] = 0.0
        if host_category_idx is not None:
            scores[zero, host_category_idx] = 1.0
        else:
            scores[zero] = 1.0
    probs = scores / scores.sum(axis=1, keepdims=True)
    return probs.astype(np.float32)


def _state_paths(output_dir: str | os.PathLike) -> dict[str, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    return {
        'reals': root / 'domain_realizations.npy',
        'state': root / 'domain_realization_state.json',
    }


def _load_state(paths: dict[str, Path], shape: tuple[int, ...]) -> tuple[np.memmap, set[int]]:
    mode = 'r+' if paths['reals'].exists() else 'w+'
    reals = np.lib.format.open_memmap(paths['reals'], mode=mode, dtype=np.int16, shape=shape)
    if mode == 'w+':
        reals[:] = -1
        reals.flush()
        paths['state'].write_text(
            json.dumps({'shape': list(shape), 'completed_realizations': [], 'status': 'running'}, indent=2),
            encoding='utf-8',
        )
        return reals, set()

    if not paths['state'].exists():
        return reals, set()
    try:
        payload = json.loads(paths['state'].read_text(encoding='utf-8'))
    except Exception:
        return reals, set()
    if tuple(payload.get('shape', [])) != tuple(shape):
        reals[:] = -1
        reals.flush()
        return reals, set()
    return reals, {int(i) for i in payload.get('completed_realizations', [])}


def _write_state(paths: dict[str, Path], shape: tuple[int, ...], completed: set[int], status: str) -> None:
    payload = {
        'updated_at': datetime.now().isoformat(timespec='seconds'),
        'shape': list(shape),
        'completed_realizations': sorted(int(i) for i in completed),
        'completed_count': len(completed),
        'status': status,
    }
    paths['state'].write_text(json.dumps(payload, indent=2), encoding='utf-8')


def simulate_categorical_domains(
    composites: pd.DataFrame,
    config: dict,
    output_dir: str,
    grid_def: dict | None = None,
) -> dict:
    """Generate stochastic domain realizations and probability grids."""
    domain_df = build_categorical_domain_data(composites, config=config, grade_field=config.get('grade_field', 'tgc_pct'))
    categories = _domain_categories(config=config)
    cat_to_id = {name: idx for idx, name in enumerate(categories)}
    domain_df['domain_id'] = domain_df['domain_group'].map(cat_to_id).astype(int)

    if grid_def is None:
        grid_def = define_grid(config, domain_df)

    grid_points = np.array(np.meshgrid(grid_def['x'], grid_def['y'], grid_def['z'], indexing='ij')).reshape(3, -1).T
    cond_points = domain_df[['x', 'y', 'z']].to_numpy(dtype=float)
    cond_ids = domain_df['domain_id'].to_numpy(dtype=int)

    cond_search_points, search_radius = _search_points(cond_points, config=config)
    grid_search_points, _ = _search_points(grid_points, config=config)
    priors = np.bincount(cond_ids, minlength=len(categories)).astype(float)
    priors = priors / max(priors.sum(), 1.0)

    sim_cfg = config.get('simulation', {}) or {}
    n_real = int(sim_cfg.get('n_real', 100))
    seed = int(sim_cfg.get('seed', 1337))
    max_neighbors = int(sim_cfg.get('max_neighbors', 24))
    prior_weight = float((config.get('domains', {}) or {}).get('prior_weight', 2.0))

    shape = (n_real, len(grid_def['x']), len(grid_def['y']), len(grid_def['z']))
    paths = _state_paths(output_dir)
    reals, completed = _load_state(paths, shape)
    pending = [i for i in range(n_real) if i not in completed]

    logger.info("Categorical domain simulation: %d realizations, %d cells", n_real, grid_points.shape[0])
    logger.info("Categorical domains: %s", categories)
    probs = _local_probabilities(
        cond_search_points=cond_search_points,
        cond_ids=cond_ids,
        grid_search_points=grid_search_points,
        search_radius=search_radius,
        max_neighbors=max_neighbors,
        priors=priors,
        prior_weight=prior_weight,
        n_categories=len(categories),
        host_category_idx=cat_to_id.get('host_waste'),
    )
    np.save(Path(output_dir) / 'domain_local_probabilities.npy', probs)

    cumulative = np.cumsum(probs, axis=1)
    for real_idx in pending:
        rng = np.random.default_rng(seed + real_idx)
        draws = rng.random(grid_points.shape[0], dtype=np.float32)
        sim_ids = np.sum(draws[:, None] > cumulative, axis=1).astype(np.int16)
        reals[real_idx] = sim_ids.reshape(shape[1:])
        if hasattr(reals, 'flush'):
            reals.flush()
        completed.add(real_idx)
        _write_state(paths, shape, completed, status='running')
        logger.info("Completed categorical domain realization %d/%d", real_idx + 1, n_real)

    _write_state(paths, shape, completed, status='completed')

    prob_dir = Path(output_dir)
    prob_dir.mkdir(parents=True, exist_ok=True)
    probability_paths = {}
    summary_rows = []
    for name, cat_id in cat_to_id.items():
        prob = (reals == cat_id).mean(axis=0).astype(np.float32)
        path = prob_dir / f'domain_probability_{name}.npy'
        np.save(path, prob)
        probability_paths[name] = str(path)
        summary_rows.append(
            {
                'domain_group': name,
                'category_id': int(cat_id),
                'mean_probability': float(prob.mean()),
                'conditioning_samples': int((cond_ids == cat_id).sum()),
            }
        )

    pd.DataFrame(summary_rows).to_csv(prob_dir / 'domain_probability_summary.csv', index=False)
    (prob_dir / 'domain_codes.json').write_text(json.dumps({'categories': categories, 'cat_to_id': cat_to_id}, indent=2), encoding='utf-8')

    return {
        'realizations': reals,
        'domain_df': domain_df,
        'grid_def': grid_def,
        'categories': categories,
        'cat_to_id': cat_to_id,
        'probability_paths': probability_paths,
        'state_path': str(paths['state']),
        'realizations_path': str(paths['reals']),
    }
