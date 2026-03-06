"""
07_sgs.py - Sequential Gaussian Simulation

Performs SGS on a 3D grid using gstools.
Correctly implements sequential conditioning for the entire grid.
"""

import sys
import os
import numpy as np
import pandas as pd
import logging
import time
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from typing import Optional

logger = logging.getLogger(__name__)

# Parallel processing
try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    logger.warning("joblib not available, using sequential processing")

# Try to import tqdm for progress bar
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    logger.warning("tqdm not available, using text-based progress")

# Import gstools
try:
    import gstools as gs
    GSTOOLS_AVAILABLE = True
except ImportError:
    GSTOOLS_AVAILABLE = False
    logger.error("gstools is required for SGS")


def validate_variogram_for_sgs(vario_model, grid_def, config):
    """Validate variogram parameters for SGS."""
    warnings = []
    range_m = vario_model.len_scale
    dx = grid_def['dx']
    dy = grid_def['dy']
    dz = grid_def['dz']
    nx, ny, nz = grid_def['nx'], grid_def['ny'], grid_def['nz']

    extent_x = nx * dx
    extent_y = ny * dy
    extent_z = nz * dz
    max_extent = max(extent_x, extent_y, extent_z)

    if range_m < max_extent * 0.05:
        warnings.append(f"WARNING: Variogram range ({range_m:.1f}m) is very small relative to grid extent")
    elif range_m > max_extent * 10:
        warnings.append(f"WARNING: Variogram range ({range_m:.1f}m) is extremely large relative to grid extent")

    total_var = vario_model.nugget + vario_model.var
    if total_var > 0:
        nugget_ratio = vario_model.nugget / total_var
        if nugget_ratio > 0.8:
            warnings.append(f"WARNING: High nugget effect ({nugget_ratio:.1%}) may result in noisy simulations")
        if vario_model.var < 0.01:
            warnings.append(f"WARNING: Sill variance is very small ({vario_model.var:.4f})")

    if range_m > 10000:
        warnings.append(f"ERROR: Variogram range ({range_m:.1f}m) is unrealistically large!")
    if range_m < 10:
        warnings.append(f"ERROR: Variogram range ({range_m:.1f}m) is unrealistically small!")

    return warnings


def validate_conditioning_data(df):
    """Validate conditioning data before SGS."""
    required_cols = {'x', 'y', 'z', 'tgc_pct'}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing conditioning columns: {sorted(missing)}")

    if df[['x', 'y', 'z', 'tgc_pct']].isna().any().any():
        raise ValueError("Conditioning data contains NaN values")

    if len(df) < 10:
        raise ValueError("Insufficient conditioning data (<10 samples)")


def define_grid(config, domain_data):
    """Define simulation grid from config and data bounds."""
    grid_config = config.get('grid', {})
    data_config = config.get('simulation', {})

    origin_xyz = grid_config.get('origin_xyz')
    nx = grid_config.get('nx')
    ny = grid_config.get('ny')
    nz = grid_config.get('nz')
    dx = grid_config.get('dx', 25)
    dy = grid_config.get('dy', 25)
    dz = grid_config.get('dz', 5)

    if origin_xyz and nx and ny and nz:
        x0, y0, z0 = origin_xyz
        x_ax = x0 + np.arange(nx) * dx
        y_ax = y0 + np.arange(ny) * dy
        z_ax = z0 + np.arange(nz) * dz

        grid_def = {
            'x': x_ax, 'y': y_ax, 'z': z_ax,
            'dx': dx, 'dy': dy, 'dz': dz,
            'nx': len(x_ax), 'ny': len(y_ax), 'nz': len(z_ax)
        }

        logger.info(
            "Grid (config): %dx%dx%d = %s cells",
            grid_def['nx'], grid_def['ny'], grid_def['nz'],
            f"{grid_def['nx']*grid_def['ny']*grid_def['nz']:,}"
        )
        return grid_def

    buffer = data_config.get('grid_buffer_m', 50)
    min_x = domain_data['x'].min() - buffer
    max_x = domain_data['x'].max() + buffer
    min_y = domain_data['y'].min() - buffer
    max_y = domain_data['y'].max() + buffer
    min_z = domain_data['z'].min() - 10
    max_z = domain_data['z'].max() + 10

    min_x = np.floor(min_x / dx) * dx
    min_y = np.floor(min_y / dy) * dy
    min_z = np.floor(min_z / dz) * dz

    x_ax = np.arange(min_x, max_x, dx)
    y_ax = np.arange(min_y, max_y, dy)
    z_ax = np.arange(min_z, max_z, dz)

    grid_def = {
        'x': x_ax, 'y': y_ax, 'z': z_ax,
        'dx': dx, 'dy': dy, 'dz': dz,
        'nx': len(x_ax), 'ny': len(y_ax), 'nz': len(z_ax)
    }

    logger.info(f"Grid: {grid_def['nx']}x{grid_def['ny']}x{grid_def['nz']} = {grid_def['nx']*grid_def['ny']*grid_def['nz']:,} cells")

    return grid_def


def run_single_realization(real_idx, seed, cond_pos, cond_val, x_ax, y_ax, z_ax,
                           vario_model, nst, chunk_size=None, search_radius=None,
                           max_neighbors=None, min_neighbors=None):
    """
    Run a single SGS realization with proper sequential conditioning.

    This simulates the ENTIRE 3D grid in one sequential pass, ensuring
    each node is conditioned on ALL previously simulated nodes.

    Parameters:
    -----------
    real_idx : int
        Realization index
    seed : int
        Random seed base
    cond_pos : tuple
        Conditioning data positions (x, y, z)
    cond_val : array
        Conditioning data values (normal scores)
    x_ax, y_ax, z_ax : arrays
        Grid axes
    vario_model : gstools variogram model
        Fitted variogram model
    nst : NormalScoreTransform
        For back-transformation
    Returns:
    --------
    real_idx : int
        Realization index
    real_data : ndarray
        Back-transformed realization (nx, ny, nz)
    """
    # If neighborhood constraints are set, use local SGS implementation
    if search_radius is not None or max_neighbors is not None or min_neighbors is not None:
        grid_points = np.array(
            np.meshgrid(x_ax, y_ax, z_ax, indexing='ij')
        ).reshape(3, -1).T
        cond_points = np.vstack(cond_pos).T
        cond_points = _build_anisometric_points(vario_model, cond_points.T)
        grid_points = _build_anisometric_points(vario_model, grid_points.T)

        idx, sim_vals, stats = run_single_realization_local(
            real_idx,
            seed,
            cond_points,
            cond_val,
            grid_points,
            vario_model,
            nst,
            search_radius=search_radius,
            max_neighbors=max_neighbors,
            min_neighbors=min_neighbors,
        )
        real_data = sim_vals.reshape(len(x_ax), len(y_ax), len(z_ax))
        if stats:
            logger.info(
                "Realization %d: avg neighbors %.1f, fallback %.1f%%",
                real_idx + 1,
                stats.get('avg_neighbors', 0.0),
                stats.get('fallback_pct', 0.0),
            )
        return idx, real_data

    # Default to global conditioning with gstools
    krige = gs.Krige(vario_model, cond_pos=cond_pos, cond_val=cond_val)
    srf = gs.CondSRF(krige)
    if chunk_size:
        field = srf.structured((x_ax, y_ax, z_ax), seed=seed + real_idx, chunk_size=chunk_size)
    else:
        field = srf.structured((x_ax, y_ax, z_ax), seed=seed + real_idx)

    real_data = nst.back_transform(field)
    return real_idx, real_data.astype(np.float32)


def _build_anisometric_points(vario_model, coords):
    coords = np.asarray(coords)
    if coords.ndim == 2 and coords.shape[0] == 3:
        return vario_model.anisometrize(coords).T
    raise ValueError("Expected coords as (3, n) array")


def _krige_local(vario_model, neighbor_pts, neighbor_vals, target_pt, jitter=1e-10):
    dist_mat = cdist(neighbor_pts, neighbor_pts)
    cov_mat = vario_model.covariance(dist_mat)
    cov_mat.flat[:: cov_mat.shape[0] + 1] += jitter

    dist_vec = cdist(neighbor_pts, target_pt[None, :]).ravel()
    cov_vec = vario_model.covariance(dist_vec)

    try:
        weights = np.linalg.solve(cov_mat, cov_vec)
    except np.linalg.LinAlgError:
        weights = np.linalg.lstsq(cov_mat, cov_vec, rcond=None)[0]

    mean = float(np.dot(weights, neighbor_vals))
    variance = float(vario_model.sill - np.dot(weights, cov_vec))
    if variance < 0:
        variance = 0.0
    return mean, variance


def _select_neighbors(tree, points, target_pt, search_radius, max_neighbors):
    if search_radius is not None:
        idx = tree.query_ball_point(target_pt, r=search_radius)
        if not idx:
            return np.array([], dtype=int)
        idx = np.array(idx, dtype=int)
        if max_neighbors is not None and len(idx) > max_neighbors:
            dists = np.linalg.norm(points[idx] - target_pt, axis=1)
            idx = idx[np.argsort(dists)[:max_neighbors]]
        return idx

    if max_neighbors is None:
        return np.arange(points.shape[0], dtype=int)

    dists, idx = tree.query(target_pt, k=min(max_neighbors, points.shape[0]))
    idx = np.atleast_1d(idx)
    return idx.astype(int)


def _parse_search_radius(search_radius) -> Optional[float]:
    """Convert configured search radius to a scalar in KDTree space.

    KDTree uses anisometrized coordinates (rotation + anisotropy already applied).
    If search_radius_m is [Rmaj,Rmid,Rmin], the KDTree radius is Rmaj.
    """
    if search_radius is None:
        return None
    if isinstance(search_radius, (list, tuple, np.ndarray)):
        r = np.array(search_radius, dtype=float).reshape(-1)
        if r.size != 3:
            raise ValueError("search_radius_m must be scalar or length-3 [Rmaj,Rmid,Rmin]")
        if (r <= 0).any():
            raise ValueError(f"search_radius_m must be positive, got {r.tolist()}")
        return float(r[0])
    r = float(search_radius)
    if r <= 0:
        raise ValueError(f"search_radius_m must be positive, got {r}")
    return r


def run_single_realization_local(
    real_idx,
    seed,
    cond_pos,
    cond_val,
    grid_points,
    vario_model,
    nst,
    search_radius=None,
    max_neighbors=None,
    min_neighbors=None,
    update_every=200,
    log_every=2000,
    jitter=1e-10,
    expand_factor=1.5,
    expand_max_steps=4,
):
    rng = np.random.default_rng(seed + real_idx)

    known_points = cond_pos.copy()
    known_vals = cond_val.copy()

    tree = cKDTree(known_points)
    base_radius = _parse_search_radius(search_radius)

    order = rng.permutation(grid_points.shape[0])
    sim_vals = np.zeros(grid_points.shape[0], dtype=float)
    new_points = []
    new_vals = []

    total_nodes = grid_points.shape[0]
    fallback_hits = 0
    expand_hits = 0
    neighbor_counts = []
    for idx, node_idx in enumerate(order):
        target_pt = grid_points[node_idx]
        r_use = base_radius
        neighbor_idx = _select_neighbors(tree, known_points, target_pt, r_use, max_neighbors)

        if min_neighbors is not None and neighbor_idx.size < min_neighbors:
            if r_use is not None:
                for _ in range(expand_max_steps):
                    r_use *= expand_factor
                    neighbor_idx = _select_neighbors(tree, known_points, target_pt, r_use, max_neighbors)
                    if neighbor_idx.size >= min_neighbors:
                        expand_hits += 1
                        break

            if neighbor_idx.size < min_neighbors and known_points.shape[0] > 0:
                dists = np.linalg.norm(known_points - target_pt, axis=1)
                neighbor_idx = np.argsort(dists)[:min(min_neighbors, known_points.shape[0])]
                fallback_hits += 1

        if neighbor_idx.size == 0:
            mean = 0.0
            variance = vario_model.sill
        else:
            neighbor_pts = known_points[neighbor_idx]
            neighbor_vals = known_vals[neighbor_idx]
            mean, variance = _krige_local(vario_model, neighbor_pts, neighbor_vals, target_pt, jitter=jitter)

        neighbor_counts.append(neighbor_idx.size)

        sim_val = mean + rng.normal() * np.sqrt(max(variance, 0.0))
        sim_vals[node_idx] = sim_val

        new_points.append(target_pt)
        new_vals.append(sim_val)

        if log_every and (idx + 1) % log_every == 0:
            logger.info(
                "Realization %d: %d/%d nodes simulated",
                real_idx + 1,
                idx + 1,
                total_nodes,
            )

        if (idx + 1) % update_every == 0:
            known_points = np.vstack([known_points, np.array(new_points)])
            known_vals = np.concatenate([known_vals, np.array(new_vals)])
            tree = cKDTree(known_points)
            new_points = []
            new_vals = []

    if new_points:
        known_points = np.vstack([known_points, np.array(new_points)])
        known_vals = np.concatenate([known_vals, np.array(new_vals)])

    # Back-transform from normal scores to original units
    real_data = nst.back_transform(sim_vals)
    return real_idx, real_data.astype(np.float32), {
        'avg_neighbors': float(np.mean(neighbor_counts)) if neighbor_counts else 0.0,
        'fallback_pct': float(fallback_hits / max(1, len(order))) * 100,
        'expand_pct': float(expand_hits / max(1, len(order))) * 100,
    }


def run_sgs(domain_data, grid_def, vario_model, nst, n_realizations=100, seed=1337, n_jobs=-1,
            chunk_size=None, search_radius=None, max_neighbors=None, min_neighbors=None, config=None):
    """
    Run Sequential Gaussian Simulation with proper sequential conditioning.

    CRITICAL: Each realization must process ALL grid nodes in a single
    sequential pass to maintain spatial continuity. Parallel processing
    is done ACROSS realizations, not within a single realization.

    Parameters:
    -----------
    domain_data : DataFrame
        Domain data with x, y, z, tgc_ns columns
    grid_def : dict
        Grid definition with x, y, z axes and dimensions
    vario_model : gstools variogram model
        Fitted variogram model
    nst : NormalScoreTransform
        For back-transformation
    n_realizations : int
        Number of realizations
    seed : int
        Random seed
    n_jobs : int
        Number of parallel jobs (-1 for all cores)

    Returns:
    --------
    dict with realizations and metadata
    """
    if not GSTOOLS_AVAILABLE:
        raise RuntimeError("gstools is required for SGS")

    # Extract conditioning data
    cond_pos = (domain_data['x'].values, domain_data['y'].values, domain_data['z'].values)
    cond_val = domain_data['tgc_ns'].values

    x_ax = grid_def['x']
    y_ax = grid_def['y']
    z_ax = grid_def['z']

    nx, ny, nz = len(x_ax), len(y_ax), len(z_ax)
    n_cells = nx * ny * nz

    # Initialize output array
    full_reals = np.zeros((n_realizations, nx, ny, nz), dtype=np.float32)

    # Set up parallel processing
    if n_jobs == -1 and JOBLIB_AVAILABLE:
        import multiprocessing
        n_jobs = min(multiprocessing.cpu_count(), n_realizations)
    elif not JOBLIB_AVAILABLE:
        n_jobs = 1

    logger.info(f"Running {n_realizations} SGS realizations")
    logger.info(f"Grid dimensions: {nx} x {ny} x {nz} = {n_cells:,} cells")
    logger.info(f"Conditioning data: {len(cond_val)} samples")
    logger.info(f"Parallel jobs: {n_jobs if JOBLIB_AVAILABLE else 1}")
    if search_radius is not None:
        logger.info(f"Configured search radius: {search_radius}")
    if max_neighbors is not None or min_neighbors is not None:
        logger.info(f"Configured neighbors: min={min_neighbors}, max={max_neighbors}")
    if chunk_size:
        logger.info(f"Kriging chunk size: {chunk_size}")

    print(f"\n>>> SGS STARTING: {n_realizations} realizations, {n_cells:,} cells", flush=True)
    print(f">>> Using {'parallel' if JOBLIB_AVAILABLE and n_jobs > 1 else 'sequential'} processing", flush=True)

    start_time = time.time()

    if JOBLIB_AVAILABLE and n_jobs > 1:
        # Parallel execution across realizations
        logger.info(f"Running {n_realizations} realizations in parallel...")

        results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(run_single_realization)(
                i, seed, cond_pos, cond_val, x_ax, y_ax, z_ax, vario_model, nst, chunk_size,
                search_radius=search_radius, max_neighbors=max_neighbors, min_neighbors=min_neighbors
            )
            for i in tqdm(range(n_realizations), desc="SGS", unit="real", disable=not TQDM_AVAILABLE)
        )

        for idx, real_data in results:
            full_reals[idx] = real_data

        logger.info("Parallel processing complete!")
    else:
        # Sequential execution
        logger.info("Running realizations sequentially...")

        if TQDM_AVAILABLE:
            pbar = tqdm(range(n_realizations), desc="SGS", unit="real")
        else:
            pbar = range(n_realizations)

        for i in pbar:
            idx, real_data = run_single_realization(
                i, seed, cond_pos, cond_val, x_ax, y_ax, z_ax, vario_model, nst, chunk_size,
                search_radius=search_radius, max_neighbors=max_neighbors, min_neighbors=min_neighbors
            )
            full_reals[idx] = real_data

            if not TQDM_AVAILABLE and (i + 1) % 10 == 0:
                logger.info(f"  Completed {i+1}/{n_realizations}")

    total_elapsed = time.time() - start_time
    logger.info("-" * 60)
    logger.info(f"SGS Complete: {n_realizations} realizations in {total_elapsed/60:.1f} minutes")
    logger.info(f"Average time per realization: {total_elapsed/n_realizations:.1f} seconds")
    logger.info(f"Output shape: {full_reals.shape}")

    if config is not None:
        trend_cfg = config.get("trend", {}) or {}
        trend_cols = trend_cfg.get("columns") or []
        coeffs = trend_cfg.get("coeffs") or []
        if trend_cfg.get("enabled") and trend_cols and coeffs:
            coeffs = np.asarray(coeffs, dtype=float).reshape(-1)
            X, Y, Z = np.meshgrid(x_ax, y_ax, z_ax, indexing="ij")
            col_map = {"x": X, "y": Y, "z": Z}
            trend_grid = np.full(X.shape, coeffs[0], dtype=float)
            for i, col in enumerate(trend_cols, start=1):
                if col not in col_map:
                    raise ValueError(f"Trend column '{col}' not in ['x','y','z'] grid map")
                if i >= len(coeffs):
                    raise ValueError("Trend coeffs length does not match columns")
                trend_grid += coeffs[i] * col_map[col]
            full_reals = full_reals + trend_grid[None, :, :, :].astype(full_reals.dtype)
            logger.info(
                "Added trend back to realizations (cols=%s, coeffs=%s)",
                trend_cols, [float(c) for c in coeffs]
            )

    return {
        'realizations': full_reals,
        'x': x_ax, 'y': y_ax, 'z': z_ax,
        'model': vario_model,
        'grid_def': grid_def,
        'timing': {'total_seconds': total_elapsed, 'avg_per_real': total_elapsed/n_realizations}
    }


def run(data_path=None, data_dir='data', config=None, output_dir='outputs/grids'):
    """Run SGS simulation."""
    if not GSTOOLS_AVAILABLE:
        raise RuntimeError("gstools is required for SGS")

    if data_path:
        df = pd.read_csv(data_path)
    else:
        from .normal_score import run as run_nst
        _, df = run_nst(data_dir=data_dir, config=config)

    validate_conditioning_data(df)

    from .variography import run as run_variography
    vario_model, _, _ = run_variography(data_path=data_path, data_dir=data_dir, config=config)

    from .normal_score import NormalScoreTransform
    nst = NormalScoreTransform()

    if 'tgc_ns' not in df.columns:
        weights = df['decluster_weight'].values if 'decluster_weight' in df.columns else None
        nst.fit(df['tgc_pct'].values, weights)
    else:
        nst.fit(df['tgc_pct'].values, df['decluster_weight'].values if 'decluster_weight' in df.columns else None)

    sim_config = config.get('simulation', {}) if config else {}
    n_real = sim_config.get('n_real', 100)
    seed = sim_config.get('seed', 1337)

    grid_def = define_grid(config, df)

    logger.info("Validating variogram parameters...")
    validation_warnings = validate_variogram_for_sgs(vario_model, grid_def, config)
    for w in validation_warnings:
        if 'ERROR' in w:
            logger.error(w)
        else:
            logger.warning(w)
    if any('ERROR' in w for w in validation_warnings):
        logger.error("Variogram validation failed!")
        raise RuntimeError("Invalid variogram parameters")

    # Get number of parallel jobs from config
    n_jobs = sim_config.get('n_jobs', -1)
    if n_jobs == -1 and JOBLIB_AVAILABLE:
        import multiprocessing
        n_jobs = min(multiprocessing.cpu_count(), n_real)

    # Reduce parallelism for large grids to limit memory pressure
    n_cells = grid_def['nx'] * grid_def['ny'] * grid_def['nz']
    if n_cells >= 200_000:
        n_jobs = min(n_jobs, 4)

    # Chunk kriging calls to avoid massive allocations
    chunk_size = sim_config.get('krige_chunk_size', 5000)
    if chunk_size is not None and chunk_size <= 0:
        chunk_size = None

    search_radius = sim_config.get('search_radius_m')
    max_neighbors = sim_config.get('max_neighbors')
    min_neighbors = sim_config.get('min_neighbors')

    result = run_sgs(
        df,
        grid_def,
        vario_model,
        nst,
        n_realizations=n_real,
        seed=seed,
        n_jobs=n_jobs,
        chunk_size=chunk_size,
        search_radius=search_radius,
        max_neighbors=max_neighbors,
        min_neighbors=min_neighbors,
        config=config,
    )

    os.makedirs(output_dir, exist_ok=True)
    from .utils.io import save_grid
    save_grid(result, output_dir, prefix='sgs')
    # Save residual/trend metadata if present
    if 'trend' in df.columns:
        meta_path = os.path.join(output_dir, 'trend_meta.json')
        import json
        trend_cols = config.get('trend', {}).get('columns', []) if config else []
        coeffs = config.get('trend', {}).get('coeffs', []) if config else []
        with open(meta_path, 'w') as f:
            json.dump({'columns': trend_cols, 'coeffs': coeffs}, f, indent=2)
    # Save SGS metadata
    meta = {
        'dx': grid_def['dx'], 'dy': grid_def['dy'], 'dz': grid_def['dz'],
        'nx': grid_def['nx'], 'ny': grid_def['ny'], 'nz': grid_def['nz'],
        'x_min': float(grid_def['x'][0]), 'x_max': float(grid_def['x'][-1]),
        'y_min': float(grid_def['y'][0]), 'y_max': float(grid_def['y'][-1]),
        'z_min': float(grid_def['z'][0]), 'z_max': float(grid_def['z'][-1]),
        'n_realizations': n_real,
        'seed': seed,
        'orebody': config.get('orebody', {}) if config else {},
    }
    import json
    with open(os.path.join(output_dir, 'sgs_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Saved {n_real} realizations to {output_dir}")

    return result
