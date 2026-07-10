"""
Cascade grade SGS conditioned on stochastic categorical domain realizations.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import os
from pathlib import Path

import numpy as np

from . import declustering, normal_score, sgs, variography
from .domains import canonical_domain_groups, split_domain_groups

logger = logging.getLogger(__name__)


def _trend_grid_flat(config: dict, grid_def: dict) -> np.ndarray | None:
    trend_cfg = config.get('trend', {}) or {}
    trend_cols = trend_cfg.get('columns') or []
    coeffs = trend_cfg.get('coeffs') or []
    if not trend_cfg.get('enabled') or not trend_cols or not coeffs:
        return None

    coeffs = np.asarray(coeffs, dtype=float).reshape(-1)
    X, Y, Z = np.meshgrid(grid_def['x'], grid_def['y'], grid_def['z'], indexing='ij')
    col_map = {'x': X, 'y': Y, 'z': Z}
    trend_grid = np.full(X.shape, coeffs[0], dtype=float)
    for i, col in enumerate(trend_cols, start=1):
        if col not in col_map:
            raise ValueError(f"Unsupported trend column '{col}' in cascade SGS")
        if i >= len(coeffs):
            raise ValueError("Trend coefficients do not match configured trend columns")
        trend_grid += coeffs[i] * col_map[col]
    return trend_grid.reshape(-1).astype(np.float32)


def prepare_domain_grade_models(composites, config: dict, output_dir: str, grade_field='tgc_pct') -> dict:
    """Prepare one grade model per categorical domain."""
    groups = split_domain_groups(composites, config=config, grade_field=grade_field)
    grid_def = sgs.define_grid(config, composites)
    models = {}
    ordered_names = [name for name in canonical_domain_groups(config=config).keys()]

    for cat_id, name in enumerate(ordered_names):
        payload = groups[name]
        branch_cfg = json.loads(json.dumps(config))
        branch_cfg['target_lith_codes'] = list(payload.get('lith_codes', []))
        branch_dir = Path(output_dir) / 'domains' / name
        branch_dir.mkdir(parents=True, exist_ok=True)
        figures_dir = branch_dir / 'figures'
        figures_dir.mkdir(parents=True, exist_ok=True)

        domain_df = payload['data'].copy()
        if domain_df.empty:
            raise RuntimeError(f"Category '{name}' has no conditioning samples for grade SGS")

        declustered, dc_stats = declustering.decluster_data(
            domain_df,
            cell_size_xy=config.get('declustering', {}).get('cell_size_xy_m', 200),
            cell_size_z=config.get('declustering', {}).get('cell_size_z_m', 5),
            grade_field=grade_field,
        )
        declustered_path = branch_dir / 'declustered.csv'
        declustered.to_csv(declustered_path, index=False)

        nst, nst_data = normal_score.run(
            data_path=str(declustered_path),
            grade_field=grade_field,
            output_path=str(branch_dir / 'nst_data.csv'),
            config=branch_cfg,
        )
        nst.save(branch_dir / 'nst_params.json')

        vario_model, _exp_variograms, ranges = variography.run(
            data_path=str(branch_dir / 'nst_data.csv'),
            config=branch_cfg,
            output_dir=str(figures_dir),
        )

        cond_points = nst_data[['x', 'y', 'z']].to_numpy(dtype=float)
        cond_cov_points = sgs._build_anisometric_points(vario_model, cond_points.T)
        cond_search_points, search_radius_local, search_meta = sgs._build_search_points(
            cond_points,
            branch_cfg.get('simulation', {}).get('search_radius_m'),
            branch_cfg,
        )

        trend_flat = _trend_grid_flat(branch_cfg, grid_def)
        (branch_dir / 'model_summary.json').write_text(
            json.dumps(
                {
                    'category_id': int(cat_id),
                    'category_name': name,
                    'n_samples': int(len(domain_df)),
                    'declustering': dc_stats,
                    'ranges_m': {k: float(v) for k, v in ranges.items()},
                    'search_meta': search_meta,
                },
                indent=2,
            ),
            encoding='utf-8',
        )

        models[name] = {
            'category_id': int(cat_id),
            'name': name,
            'config': branch_cfg,
            'domain_df': domain_df,
            'nst': nst,
            'vario_model': vario_model,
            'cond_val': nst_data['tgc_ns'].to_numpy(dtype=float),
            'cond_cov_points': cond_cov_points,
            'cond_search_points': cond_search_points,
            'search_radius': search_radius_local,
            'search_meta': search_meta,
            'min_neighbors': int(branch_cfg.get('simulation', {}).get('min_neighbors', 8)),
            'max_neighbors': int(branch_cfg.get('simulation', {}).get('max_neighbors', 24)),
            'update_every': int(branch_cfg.get('simulation', {}).get('local_update_every', 1)),
            'require_full_neighborhood': bool(branch_cfg.get('simulation', {}).get('require_full_neighborhood', False)),
            'allow_zero_neighbor_fallback': bool(branch_cfg.get('simulation', {}).get('cascade_allow_zero_neighbor_fallback', True)),
            'seed_base': int(branch_cfg.get('simulation', {}).get('seed', 1337)) + int(cat_id) * 100000,
            'trend_flat': trend_flat,
            'figures_dir': str(figures_dir),
            'ranges': ranges,
        }

    return {'models': models, 'grid_def': grid_def}


def _simulate_cascade_realization(
    ridx: int,
    domain_realizations: np.ndarray,
    models: dict,
    grid_points_full: np.ndarray,
    n_cells: int,
    checkpoint_every: int,
    output_dir: str,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Simulate one cascade realization across all categorical domains."""
    domain_flat = np.asarray(domain_realizations[ridx]).reshape(-1)
    full_grade_flat = np.full(n_cells, np.nan, dtype=np.float32)
    full_ns_flat = np.full(n_cells, np.nan, dtype=np.float32)

    for name, model in models.items():
        cat_id = int(model['category_id'])
        mask_flat = domain_flat == cat_id
        progress_key = f"cascade_{name}_real_{ridx:04d}"
        if not np.any(mask_flat):
            sgs._clear_local_progress(output_dir, progress_key)
            continue

        grid_points = grid_points_full[mask_flat]
        grid_cov_points = sgs._build_anisometric_points(model['vario_model'], grid_points.T)
        grid_search_points, _, _ = sgs._build_search_points(
            grid_points,
            model['config'].get('simulation', {}).get('search_radius_m'),
            model['config'],
        )
        _, real_data, real_ns, stats = sgs.run_single_realization_local(
            ridx,
            model['seed_base'],
            model['cond_search_points'],
            model['cond_cov_points'],
            model['cond_val'],
            grid_search_points,
            grid_cov_points,
            model['vario_model'],
            model['nst'],
            search_radius=model['search_radius'],
            max_neighbors=model['max_neighbors'],
            min_neighbors=model['min_neighbors'],
            update_every=model['update_every'],
            checkpoint_every=checkpoint_every,
            require_full_neighborhood=model['require_full_neighborhood'],
            allow_zero_neighbor_fallback=model['allow_zero_neighbor_fallback'],
            output_dir=output_dir,
            progress_key=progress_key,
        )
        if model['trend_flat'] is not None:
            real_data = real_data + model['trend_flat'][mask_flat]
        full_grade_flat[mask_flat] = real_data.astype(np.float32)
        full_ns_flat[mask_flat] = real_ns.astype(np.float32)
        logger.info(
            "Cascade realization %d domain %s: active=%d avg_neighbors=%.1f under-min=%.1f%% zero-neighbor=%.3f%%",
            ridx + 1,
            name,
            int(mask_flat.sum()),
            float(stats.get('avg_neighbors', 0.0)),
            float(stats.get('under_min_neighbors_pct', 0.0)),
            float(stats.get('zero_neighbor_pct', 0.0)),
        )

    if np.isnan(full_grade_flat).any() or np.isnan(full_ns_flat).any():
        raise RuntimeError(f"Cascade realization {ridx + 1} contains unassigned cells after domain-wise SGS")

    return ridx, full_grade_flat, full_ns_flat


def simulate_cascade_grades(
    domain_realizations: np.ndarray,
    models: dict,
    grid_def: dict,
    config: dict,
    output_dir: str,
) -> dict:
    """Simulate grades realization-by-realization inside each categorical domain."""
    n_real = int(domain_realizations.shape[0])
    nx, ny, nz = int(domain_realizations.shape[1]), int(domain_realizations.shape[2]), int(domain_realizations.shape[3])
    grid_shape = (nx, ny, nz)
    n_cells = int(nx * ny * nz)
    grid_points_full = np.array(np.meshgrid(grid_def['x'], grid_def['y'], grid_def['z'], indexing='ij')).reshape(3, -1).T

    checkpoint_shape = (n_real, nx, ny, nz)
    full_reals, full_reals_ns, completed, checkpoint_paths = sgs._init_checkpoint_arrays(output_dir, checkpoint_shape)
    sim_cfg = config.get('simulation', {}) or {}
    checkpoint_every = int(sim_cfg.get('checkpoint_every', 5000))
    requested_cascade_n_jobs = max(1, int(sim_cfg.get('cascade_n_jobs', 1) or 1))
    safe_default_cap = 1 if n_cells >= 1_000_000 else requested_cascade_n_jobs
    safe_cascade_cap = max(1, int(sim_cfg.get('cascade_safe_max_workers', safe_default_cap) or safe_default_cap))
    cascade_n_jobs = min(requested_cascade_n_jobs, safe_cascade_cap)
    if cascade_n_jobs < requested_cascade_n_jobs:
        logger.warning(
            "Requested %d cascade SGS workers on %d-cell grid; using %d active worker(s) to avoid memory-pressure termination.",
            requested_cascade_n_jobs,
            n_cells,
            cascade_n_jobs,
        )
    pending = [i for i in range(n_real) if i not in completed]
    if pending:
        logger.info("Cascade grade SGS: %d pending realizations", len(pending))
        logger.info(
            "Cascade grade SGS realization workers: %d",
            min(cascade_n_jobs, len(pending)),
        )

    def _commit_realization(ridx: int, full_grade_flat: np.ndarray, full_ns_flat: np.ndarray) -> None:
        sgs._store_realization_checkpoint(
            full_reals,
            full_reals_ns,
            checkpoint_paths,
            checkpoint_shape,
            completed,
            ridx,
            full_grade_flat,
            full_ns_flat,
        )
        for name in models:
            sgs._clear_local_progress(output_dir, f"cascade_{name}_real_{ridx:04d}")
        logger.info("Completed cascade grade realization %d/%d", ridx + 1, n_real)

    parallel_completed = False
    if cascade_n_jobs > 1 and len(pending) > 1:
        try:
            future_map = {}
            with ThreadPoolExecutor(
                max_workers=min(cascade_n_jobs, len(pending)),
                thread_name_prefix="cascade",
            ) as executor:
                for ridx in pending:
                    future = executor.submit(
                        _simulate_cascade_realization,
                        ridx,
                        domain_realizations,
                        models,
                        grid_points_full,
                        n_cells,
                        checkpoint_every,
                        output_dir,
                    )
                    future_map[future] = ridx

                iterator = as_completed(future_map)
                if sgs.TQDM_AVAILABLE:
                    iterator = sgs.tqdm(iterator, total=len(future_map), desc="Cascade SGS", unit="real")
                for future in iterator:
                    ridx, full_grade_flat, full_ns_flat = future.result()
                    _commit_realization(ridx, full_grade_flat, full_ns_flat)
            parallel_completed = True
        except Exception as exc:
            if sgs._is_memory_pressure_error(exc):
                logger.exception(
                    "Parallel cascade SGS hit memory pressure; retrying sequentially with the same scientific parameters."
                )
                cascade_n_jobs = 1
                pending = [i for i in range(n_real) if i not in completed]
            else:
                raise

    if not parallel_completed:
        pending = [i for i in range(n_real) if i not in completed]
        for ridx in pending:
            ridx, full_grade_flat, full_ns_flat = _simulate_cascade_realization(
                ridx,
                domain_realizations,
                models,
                grid_points_full,
                n_cells,
                checkpoint_every,
                output_dir,
            )
            _commit_realization(ridx, full_grade_flat, full_ns_flat)

    if checkpoint_paths is not None:
        sgs._write_checkpoint_state(checkpoint_paths, checkpoint_shape, completed, status='completed')

    total_seconds = float(n_real)  # lightweight placeholder; detailed timing is not tracked per category here
    return {
        'realizations': full_reals,
        'realizations_ns': full_reals_ns,
        'x': grid_def['x'],
        'y': grid_def['y'],
        'z': grid_def['z'],
        'grid_def': grid_def,
        'timing': {'total_seconds': total_seconds, 'avg_per_real': total_seconds / max(1, n_real)},
    }
