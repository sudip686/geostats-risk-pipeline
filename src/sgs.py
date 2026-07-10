"""
07_sgs.py - Sequential Gaussian Simulation

Performs SGS on a 3D grid using gstools.
Correctly implements sequential conditioning for the entire grid.
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import logging
import time
import tempfile
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from typing import Optional
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from .variography import build_orebody_axes, orebody_from_config

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


def validate_conditioning_data(df, grade_field='tgc_pct'):
    """Validate conditioning data before SGS."""
    required_cols = {'x', 'y', 'z', grade_field}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing conditioning columns: {sorted(missing)}")

    if df[['x', 'y', 'z', grade_field]].isna().any().any():
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
                           max_neighbors=None, min_neighbors=None, update_every=1):
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
        Back-transformed realization in grade units (nx, ny, nz)
    real_ns : ndarray
        Realization in normal-score units (nx, ny, nz)
    """
    # Default to global conditioning with gstools
    krige = gs.Krige(vario_model, cond_pos=cond_pos, cond_val=cond_val)
    srf = gs.CondSRF(krige)
    if chunk_size:
        field = srf.structured((x_ax, y_ax, z_ax), seed=seed + real_idx, chunk_size=chunk_size)
    else:
        field = srf.structured((x_ax, y_ax, z_ax), seed=seed + real_idx)

    real_data = nst.back_transform(field)
    return real_idx, real_data.astype(np.float32), field.astype(np.float32)


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


def _normalize_search_radius(search_radius) -> np.ndarray | None:
    if search_radius is None:
        return None
    r = np.array(search_radius, dtype=float).reshape(-1)
    if r.size == 0:
        return None
    if (r <= 0).any():
        raise ValueError(f"search_radius_m must be positive, got {r.tolist()}")
    return r


def _project_to_orebody_axes(coords: np.ndarray, config: dict | None) -> np.ndarray:
    orebody = orebody_from_config(config)
    if not orebody:
        raise ValueError("Ellipsoidal search_radius_m requires orebody orientation in config")

    strike_deg = orebody.get('strike_deg')
    dip_deg = orebody.get('dip_deg')
    dip_direction_deg = orebody.get('dip_direction_deg')
    dip_positive_down = bool(orebody.get('dip_positive_down', True))
    if strike_deg is None or dip_deg is None:
        raise ValueError("Ellipsoidal search_radius_m requires strike_deg and dip_deg")

    axes = build_orebody_axes(
        float(strike_deg),
        float(dip_deg),
        float(dip_direction_deg) if dip_direction_deg is not None else None,
        dip_positive_down=dip_positive_down,
    )
    basis = np.vstack([axes['strike'], axes['dip'], axes['normal']])
    return coords @ basis.T


def _build_search_points(coords: np.ndarray, search_radius, config: dict | None) -> tuple[np.ndarray, Optional[float], dict]:
    radii = _normalize_search_radius(search_radius)
    if radii is None:
        return coords.astype(float), None, {'mode': 'global', 'radii_m': None}
    if radii.size == 1:
        return coords.astype(float), float(radii[0]), {'mode': 'scalar', 'radii_m': [float(radii[0])]}
    if radii.size != 3:
        raise ValueError("search_radius_m must be scalar or length-3 [strike, dip, normal]")

    projected = _project_to_orebody_axes(coords, config)
    scaled = projected / radii.reshape(1, 3)
    return scaled.astype(float), 1.0, {'mode': 'ellipsoid', 'radii_m': radii.tolist()}


def _reshape_realization_grid(values: np.ndarray, grid_shape: tuple[int, int, int]) -> np.ndarray:
    arr = np.asarray(values)
    if arr.shape == grid_shape:
        return arr
    if arr.ndim == 1 and arr.size == int(np.prod(grid_shape)):
        return arr.reshape(grid_shape)
    raise ValueError(f"Unexpected realization shape {arr.shape}; expected {grid_shape} or flat size {int(np.prod(grid_shape))}")


def _embed_realization_grid(
    values: np.ndarray,
    grid_shape: tuple[int, int, int],
    flat_indices: np.ndarray | None = None,
) -> np.ndarray:
    arr = np.asarray(values)
    if flat_indices is None:
        return _reshape_realization_grid(arr, grid_shape)
    flat_indices = np.asarray(flat_indices, dtype=int).reshape(-1)
    if arr.ndim != 1 or arr.size != flat_indices.size:
        raise ValueError(
            f"Masked realization has shape {arr.shape}; expected flat size {flat_indices.size}"
        )
    embedded = np.full(int(np.prod(grid_shape)), np.nan, dtype=arr.dtype)
    embedded[flat_indices] = arr
    return embedded.reshape(grid_shape)


def _is_memory_pressure_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return (
        isinstance(exc, MemoryError)
        or "bad_alloc" in text
        or "paging file" in text
        or "cannot allocate memory" in text
        or "out of memory" in text
    )


def _checkpoint_paths(output_dir: str | os.PathLike | None) -> dict[str, Path] | None:
    if output_dir is None:
        return None
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    return {
        "grade": root / "sgs_reals_checkpoint.npy",
        "ns": root / "sgs_reals_ns_checkpoint.npy",
        "state": root / "sgs_checkpoint_state.json",
    }


def _load_checkpoint_state(paths: dict[str, Path], shape: tuple[int, ...]) -> dict:
    state_path = paths["state"]
    if not state_path.exists():
        return {"shape": list(shape), "completed_realizations": []}
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Checkpoint state file is unreadable; starting with a clean checkpoint.")
        return {"shape": list(shape), "completed_realizations": []}
    if tuple(state.get("shape", [])) != tuple(shape):
        logger.warning("Checkpoint shape mismatch; ignoring stale checkpoint state.")
        return {"shape": list(shape), "completed_realizations": []}
    return state


def _write_checkpoint_state(paths: dict[str, Path], shape: tuple[int, ...], completed: set[int], status: str) -> None:
    payload = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "shape": list(shape),
        "completed_realizations": sorted(int(i) for i in completed),
        "completed_count": len(completed),
        "status": status,
    }
    paths["state"].write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _init_checkpoint_arrays(
    output_dir: str | os.PathLike | None,
    shape: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, set[int], dict[str, Path] | None]:
    paths = _checkpoint_paths(output_dir)
    if paths is None:
        return np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32), set(), None

    state = _load_checkpoint_state(paths, shape)
    completed = {int(i) for i in state.get("completed_realizations", [])}

    mode = "r+" if paths["grade"].exists() and paths["ns"].exists() else "w+"
    grade = np.lib.format.open_memmap(paths["grade"], mode=mode, dtype=np.float32, shape=shape)
    ns = np.lib.format.open_memmap(paths["ns"], mode=mode, dtype=np.float32, shape=shape)

    if mode == "w+":
        grade[:] = np.nan
        ns[:] = np.nan
        grade.flush()
        ns.flush()
        _write_checkpoint_state(paths, shape, completed=set(), status="running")
    else:
        logger.info(
            "Resuming from SGS checkpoint: %d/%d realizations already completed.",
            len(completed),
            shape[0],
        )

    return grade, ns, completed, paths


def _store_realization_checkpoint(
    full_reals: np.ndarray,
    full_reals_ns: np.ndarray,
    checkpoint_paths: dict[str, Path] | None,
    checkpoint_shape: tuple[int, ...],
    completed: set[int],
    idx: int,
    real_data: np.ndarray,
    real_ns: np.ndarray,
    flat_indices: np.ndarray | None = None,
) -> None:
    full_reals[idx] = _embed_realization_grid(real_data, checkpoint_shape[1:], flat_indices=flat_indices)
    full_reals_ns[idx] = _embed_realization_grid(real_ns, checkpoint_shape[1:], flat_indices=flat_indices)
    completed.add(int(idx))
    if checkpoint_paths is not None:
        if hasattr(full_reals, "flush"):
            full_reals.flush()
        if hasattr(full_reals_ns, "flush"):
            full_reals_ns.flush()
        _write_checkpoint_state(checkpoint_paths, checkpoint_shape, completed, status="running")


def _local_progress_dir(output_dir: str | os.PathLike | None) -> Path | None:
    if output_dir is None:
        return None
    root = Path(output_dir) / "local_resume"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _local_progress_path(output_dir: str | os.PathLike | None, progress_key: str | None) -> Path | None:
    if not progress_key:
        return None
    root = _local_progress_dir(output_dir)
    if root is None:
        return None
    safe_key = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(progress_key))
    return root / f"{safe_key}.npz"


def _load_local_progress(
    output_dir: str | os.PathLike | None,
    progress_key: str | None,
    total_nodes: int,
) -> dict | None:
    path = _local_progress_path(output_dir, progress_key)
    if path is None or not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as payload:
            sim_vals = np.asarray(payload["sim_vals"], dtype=np.float32)
            next_pos = int(payload["next_pos"][()])
            under_min_hits = int(payload["under_min_hits"][()])
            zero_neighbor_hits = int(payload["zero_neighbor_hits"][()]) if "zero_neighbor_hits" in payload else 0
            neighbor_sum = float(payload["neighbor_sum"][()])
            neighbor_count = int(payload["neighbor_count"][()])
            rng_state_json = str(payload["rng_state_json"][()])
    except Exception:
        logger.warning("Local SGS progress file is unreadable for %s; ignoring partial checkpoint.", progress_key)
        return None

    if sim_vals.shape != (int(total_nodes),):
        logger.warning("Local SGS progress shape mismatch for %s; ignoring partial checkpoint.", progress_key)
        return None
    if next_pos < 0 or next_pos > int(total_nodes):
        logger.warning("Local SGS progress node count is invalid for %s; ignoring partial checkpoint.", progress_key)
        return None

    try:
        rng_state = json.loads(rng_state_json)
    except Exception:
        logger.warning("Local SGS RNG state is unreadable for %s; ignoring partial checkpoint.", progress_key)
        return None

    return {
        "sim_vals": sim_vals.astype(float, copy=False),
        "next_pos": next_pos,
        "under_min_hits": under_min_hits,
        "zero_neighbor_hits": zero_neighbor_hits,
        "neighbor_sum": neighbor_sum,
        "neighbor_count": neighbor_count,
        "rng_state": rng_state,
        "path": path,
    }


def _write_local_progress(
    output_dir: str | os.PathLike | None,
    progress_key: str | None,
    sim_vals: np.ndarray,
    next_pos: int,
    under_min_hits: int,
    zero_neighbor_hits: int,
    neighbor_sum: float,
    neighbor_count: int,
    rng_state: dict,
) -> None:
    path = _local_progress_path(output_dir, progress_key)
    if path is None:
        return
    with tempfile.NamedTemporaryFile(dir=str(path.parent), suffix=".npz", delete=False) as tmp:
        np.savez(
            tmp,
            sim_vals=np.asarray(sim_vals, dtype=np.float32),
            next_pos=np.asarray(int(next_pos), dtype=np.int64),
            under_min_hits=np.asarray(int(under_min_hits), dtype=np.int64),
            zero_neighbor_hits=np.asarray(int(zero_neighbor_hits), dtype=np.int64),
            neighbor_sum=np.asarray(float(neighbor_sum), dtype=np.float64),
            neighbor_count=np.asarray(int(neighbor_count), dtype=np.int64),
            rng_state_json=np.asarray(json.dumps(rng_state), dtype=str),
        )
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def _clear_local_progress(output_dir: str | os.PathLike | None, progress_key: str | None) -> None:
    path = _local_progress_path(output_dir, progress_key)
    if path is None:
        return
    try:
        if path.exists():
            path.unlink()
    except OSError:
        logger.warning("Failed to remove local SGS progress file for %s", progress_key)


def run_single_realization_local(
    real_idx,
    seed,
    cond_search_points,
    cond_cov_points,
    cond_val,
    grid_search_points,
    grid_cov_points,
    vario_model,
    nst,
    search_radius=None,
    max_neighbors=None,
    min_neighbors=None,
    update_every=200,
    log_every=2000,
    checkpoint_every=5000,
    jitter=1e-10,
    require_full_neighborhood=False,
    allow_zero_neighbor_fallback=False,
    output_dir=None,
    progress_key=None,
):
    rng = np.random.default_rng(seed + real_idx)
    order = rng.permutation(grid_search_points.shape[0])

    progress = _load_local_progress(output_dir, progress_key, grid_search_points.shape[0])
    if progress is not None:
        sim_vals = progress["sim_vals"]
        start_pos = int(progress["next_pos"])
        under_min_hits = int(progress["under_min_hits"])
        zero_neighbor_hits = int(progress.get("zero_neighbor_hits", 0))
        neighbor_sum = float(progress["neighbor_sum"])
        neighbor_count = int(progress["neighbor_count"])
        rng.bit_generator.state = progress["rng_state"]
        logger.info(
            "Resuming local SGS task %s at node %d/%d",
            progress_key or f"real_{real_idx + 1}",
            start_pos,
            grid_search_points.shape[0],
        )
    else:
        sim_vals = np.full(grid_search_points.shape[0], np.nan, dtype=float)
        start_pos = 0
        under_min_hits = 0
        zero_neighbor_hits = 0
        neighbor_sum = 0.0
        neighbor_count = 0
        _write_local_progress(
            output_dir,
            progress_key,
            sim_vals,
            next_pos=0,
            under_min_hits=0,
            zero_neighbor_hits=0,
            neighbor_sum=0.0,
            neighbor_count=0,
            rng_state=rng.bit_generator.state,
        )

    known_search_points = cond_search_points.copy()
    known_cov_points = cond_cov_points.copy()
    known_vals = cond_val.copy()

    if start_pos > 0:
        completed_nodes = order[:start_pos]
        known_search_points = np.vstack([known_search_points, grid_search_points[completed_nodes]])
        known_cov_points = np.vstack([known_cov_points, grid_cov_points[completed_nodes]])
        known_vals = np.concatenate([known_vals, sim_vals[completed_nodes]])

    tree = cKDTree(known_search_points)
    base_radius = search_radius

    new_search_points = []
    new_cov_points = []
    new_vals = []

    total_nodes = grid_search_points.shape[0]
    for pos in range(start_pos, total_nodes):
        node_idx = int(order[pos])
        target_search_pt = grid_search_points[node_idx]
        target_cov_pt = grid_cov_points[node_idx]
        neighbor_idx = _select_neighbors(tree, known_search_points, target_search_pt, base_radius, max_neighbors)

        if min_neighbors is not None and neighbor_idx.size < min_neighbors:
            under_min_hits += 1
            if require_full_neighborhood and neighbor_idx.size == 0 and not allow_zero_neighbor_fallback:
                raise RuntimeError(
                    f"Realization {real_idx + 1}: fixed neighborhood found no neighbors "
                    f"for a simulated node; minimum configured target is {min_neighbors}"
                )

        if neighbor_idx.size == 0:
            zero_neighbor_hits += 1
            mean = 0.0
            variance = vario_model.sill
        else:
            neighbor_pts = known_cov_points[neighbor_idx]
            neighbor_vals = known_vals[neighbor_idx]
            mean, variance = _krige_local(vario_model, neighbor_pts, neighbor_vals, target_cov_pt, jitter=jitter)

        neighbor_sum += float(neighbor_idx.size)
        neighbor_count += 1

        sim_val = mean + rng.normal() * np.sqrt(max(variance, 0.0))
        sim_vals[node_idx] = sim_val

        new_search_points.append(target_search_pt)
        new_cov_points.append(target_cov_pt)
        new_vals.append(sim_val)

        if log_every and (pos + 1) % log_every == 0:
            logger.info(
                "Realization %d: %d/%d nodes simulated",
                real_idx + 1,
                pos + 1,
                total_nodes,
            )

        if (pos + 1) % update_every == 0:
            known_search_points = np.vstack([known_search_points, np.array(new_search_points)])
            known_cov_points = np.vstack([known_cov_points, np.array(new_cov_points)])
            known_vals = np.concatenate([known_vals, np.array(new_vals)])
            tree = cKDTree(known_search_points)
            new_search_points = []
            new_cov_points = []
            new_vals = []

        if checkpoint_every and ((pos + 1) % checkpoint_every == 0 or (pos + 1) == total_nodes):
            _write_local_progress(
                output_dir,
                progress_key,
                sim_vals,
                next_pos=pos + 1,
                under_min_hits=under_min_hits,
                zero_neighbor_hits=zero_neighbor_hits,
                neighbor_sum=neighbor_sum,
                neighbor_count=neighbor_count,
                rng_state=rng.bit_generator.state,
            )

    if new_search_points:
        known_search_points = np.vstack([known_search_points, np.array(new_search_points)])
        known_cov_points = np.vstack([known_cov_points, np.array(new_cov_points)])
        known_vals = np.concatenate([known_vals, np.array(new_vals)])

    # Back-transform from normal scores to original units
    real_data = nst.back_transform(sim_vals)
    return real_idx, real_data.astype(np.float32), sim_vals.astype(np.float32), {
        'avg_neighbors': float(neighbor_sum / max(1, neighbor_count)),
        'under_min_neighbors_pct': float(under_min_hits / max(1, total_nodes)) * 100,
        'zero_neighbor_pct': float(zero_neighbor_hits / max(1, total_nodes)) * 100,
        'zero_neighbor_hits': int(zero_neighbor_hits),
        'active_nodes': int(total_nodes),
    }


def run_sgs(domain_data, grid_def, vario_model, nst, n_realizations=100, seed=1337, n_jobs=-1,
            chunk_size=None, search_radius=None, max_neighbors=None, min_neighbors=None,
            update_every=1, config=None, output_dir=None, grid_mask=None,
            require_full_neighborhood=False, checkpoint_every=5000):
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
    grid_shape = (nx, ny, nz)
    flat_indices = None
    active_cell_count = n_cells

    if grid_mask is not None:
        mask = np.asarray(grid_mask, dtype=bool)
        if mask.shape != grid_shape:
            raise ValueError(f"grid_mask shape {mask.shape} does not match grid shape {grid_shape}")
        flat_indices = np.flatnonzero(mask.reshape(-1))
        active_cell_count = int(flat_indices.size)
        if active_cell_count == 0:
            raise ValueError("grid_mask selects zero simulation cells")

    use_local_sgs = search_radius is not None or max_neighbors is not None or min_neighbors is not None

    checkpoint_shape = (n_realizations, nx, ny, nz)
    full_reals, full_reals_ns, completed, checkpoint_paths = _init_checkpoint_arrays(output_dir, checkpoint_shape)
    pending = [i for i in range(n_realizations) if i not in completed]
    if not pending:
        logger.info("All realizations already present in checkpoint; reusing existing SGS checkpoint arrays.")

    # Set up parallel processing
    if n_jobs == -1 and JOBLIB_AVAILABLE:
        import multiprocessing
        n_jobs = min(multiprocessing.cpu_count(), n_realizations)
    elif not JOBLIB_AVAILABLE:
        n_jobs = 1

    if not use_local_sgs and n_jobs != 1:
        logger.info(
            "Using sequential gstools path with reusable kriging state for fixed global simulation."
        )
        n_jobs = 1

    logger.info(f"Running {n_realizations} SGS realizations")
    logger.info(f"Grid dimensions: {nx} x {ny} x {nz} = {n_cells:,} cells")
    if flat_indices is not None:
        logger.info(f"Active masked cells: {active_cell_count:,}")
    logger.info(f"Conditioning data: {len(cond_val)} samples")
    logger.info(f"Parallel jobs: {n_jobs if JOBLIB_AVAILABLE else 1}")
    logger.info(f"Completed realizations found in checkpoint: {len(completed)}")
    if search_radius is not None:
        logger.info(f"Configured search radius: {search_radius}")
    if max_neighbors is not None or min_neighbors is not None:
        logger.info(f"Configured neighbors: min={min_neighbors}, max={max_neighbors}")
    if chunk_size:
        logger.info(f"Kriging chunk size: {chunk_size}")

    print(f"\n>>> SGS STARTING: {n_realizations} realizations, {n_cells:,} cells", flush=True)
    print(f">>> Using {'parallel' if JOBLIB_AVAILABLE and n_jobs > 1 else 'sequential'} processing", flush=True)

    start_time = time.time()

    parallel_completed = False
    if JOBLIB_AVAILABLE and n_jobs > 1 and pending:
        # Parallel execution across realizations
        logger.info(f"Running {len(pending)} pending realizations in parallel...")

        try:
            if use_local_sgs:
                grid_points_full = np.array(np.meshgrid(x_ax, y_ax, z_ax, indexing='ij')).reshape(3, -1).T
                grid_points = grid_points_full[flat_indices] if flat_indices is not None else grid_points_full
                cond_points = np.vstack(cond_pos).T
                cond_cov_points = _build_anisometric_points(vario_model, cond_points.T)
                grid_cov_points = _build_anisometric_points(vario_model, grid_points.T)
                cond_search_points, search_radius_local, search_meta = _build_search_points(cond_points, search_radius, config)
                grid_search_points, _, _ = _build_search_points(grid_points, search_radius, config)
                logger.info("Local SGS search mode: %s", search_meta)
                logger.info(
                    "Using shared-memory threads for local SGS to avoid process-copy memory pressure."
                )
                future_map = {}
                with ThreadPoolExecutor(max_workers=n_jobs, thread_name_prefix="sgs") as executor:
                    for i in pending:
                        future = executor.submit(
                            run_single_realization_local,
                            i,
                            seed,
                            cond_search_points,
                            cond_cov_points,
                            cond_val,
                            grid_search_points,
                            grid_cov_points,
                            vario_model,
                            nst,
                            search_radius=search_radius_local,
                            max_neighbors=max_neighbors,
                            min_neighbors=min_neighbors,
                            update_every=update_every,
                            checkpoint_every=checkpoint_every,
                            require_full_neighborhood=require_full_neighborhood,
                            output_dir=output_dir,
                            progress_key=f"real_{i:04d}",
                        )
                        future_map[future] = i
                    iterator = as_completed(future_map)
                    if TQDM_AVAILABLE:
                        iterator = tqdm(iterator, total=len(future_map), desc="SGS", unit="real")
                    for future in iterator:
                        idx, real_data, real_ns, stats = future.result()
                        _store_realization_checkpoint(
                            full_reals,
                            full_reals_ns,
                            checkpoint_paths,
                            checkpoint_shape,
                            completed,
                            idx,
                            real_data,
                            real_ns,
                            flat_indices=flat_indices,
                        )
                        logger.info(
                            "Completed realization %d/%d (avg neighbors %.1f, under-min %.1f%%, checkpointed %d/%d)",
                            idx + 1,
                            n_realizations,
                            stats.get('avg_neighbors', 0.0),
                            stats.get('under_min_neighbors_pct', 0.0),
                            len(completed),
                            n_realizations,
                        )
                        _clear_local_progress(output_dir, f"real_{idx:04d}")
            else:
                results = Parallel(n_jobs=n_jobs, verbose=0)(
                    delayed(run_single_realization)(
                        i, seed, cond_pos, cond_val, x_ax, y_ax, z_ax, vario_model, nst, chunk_size,
                        search_radius=search_radius, max_neighbors=max_neighbors,
                        min_neighbors=min_neighbors, update_every=update_every
                    )
                    for i in tqdm(pending, desc="SGS", unit="real", disable=not TQDM_AVAILABLE)
                )
                for idx, real_data, real_ns, *rest in results:
                    _store_realization_checkpoint(
                        full_reals,
                        full_reals_ns,
                        checkpoint_paths,
                        checkpoint_shape,
                        completed,
                        idx,
                        real_data,
                        real_ns,
                        flat_indices=flat_indices,
                    )

            logger.info("Parallel processing complete!")
            parallel_completed = True
        except Exception as exc:
            if use_local_sgs and _is_memory_pressure_error(exc):
                logger.exception(
                    "Parallel local SGS hit memory pressure; retrying sequentially with the same scientific parameters."
                )
                print(
                    ">>> Parallel SGS hit memory pressure; retrying sequentially with the same parameters",
                    flush=True,
                )
                n_jobs = 1
            else:
                raise

    if not parallel_completed and pending:
        # Sequential execution
        logger.info("Running realizations sequentially...")

        if TQDM_AVAILABLE:
            pbar = tqdm(pending, desc="SGS", unit="real")
        else:
            pbar = pending

        if use_local_sgs:
            grid_points_full = np.array(np.meshgrid(x_ax, y_ax, z_ax, indexing='ij')).reshape(3, -1).T
            grid_points = grid_points_full[flat_indices] if flat_indices is not None else grid_points_full
            cond_points = np.vstack(cond_pos).T
            cond_cov_points = _build_anisometric_points(vario_model, cond_points.T)
            grid_cov_points = _build_anisometric_points(vario_model, grid_points.T)
            cond_search_points, search_radius_local, search_meta = _build_search_points(cond_points, search_radius, config)
            grid_search_points, _, _ = _build_search_points(grid_points, search_radius, config)
            logger.info("Local SGS search mode: %s", search_meta)
            for i in pbar:
                idx, real_data, real_ns, stats = run_single_realization_local(
                    i,
                    seed,
                    cond_search_points,
                    cond_cov_points,
                    cond_val,
                    grid_search_points,
                    grid_cov_points,
                    vario_model,
                    nst,
                    search_radius=search_radius_local,
                    max_neighbors=max_neighbors,
                    min_neighbors=min_neighbors,
                    update_every=update_every,
                    checkpoint_every=checkpoint_every,
                    require_full_neighborhood=require_full_neighborhood,
                    output_dir=output_dir,
                    progress_key=f"real_{i:04d}",
                )
                _store_realization_checkpoint(
                    full_reals,
                    full_reals_ns,
                    checkpoint_paths,
                    checkpoint_shape,
                    completed,
                    idx,
                    real_data,
                    real_ns,
                    flat_indices=flat_indices,
                )
                if stats:
                    logger.info(
                        "Realization %d: avg neighbors %.1f, under-min %.1f%%, checkpointed %d/%d",
                        idx + 1,
                        stats.get('avg_neighbors', 0.0),
                        stats.get('under_min_neighbors_pct', 0.0),
                        len(completed),
                        n_realizations,
                    )
                _clear_local_progress(output_dir, f"real_{idx:04d}")

                if not TQDM_AVAILABLE and (i + 1) % 10 == 0:
                    logger.info(f"  Completed {i+1}/{n_realizations}")
        else:
            krige = gs.Krige(vario_model, cond_pos=cond_pos, cond_val=cond_val)
            srf = gs.CondSRF(krige)
            for i in pbar:
                if chunk_size:
                    field = srf.structured((x_ax, y_ax, z_ax), seed=seed + i, chunk_size=chunk_size)
                else:
                    field = srf.structured((x_ax, y_ax, z_ax), seed=seed + i)
                if flat_indices is not None:
                    field = np.asarray(field, dtype=np.float32)
                    field_flat = field.reshape(-1)
                    active_field = field_flat[flat_indices]
                    data_field = nst.back_transform(field).astype(np.float32).reshape(-1)[flat_indices]
                    ns_field = active_field.astype(np.float32)
                else:
                    data_field = nst.back_transform(field).astype(np.float32)
                    ns_field = field.astype(np.float32)
                _store_realization_checkpoint(
                    full_reals,
                    full_reals_ns,
                    checkpoint_paths,
                    checkpoint_shape,
                    completed,
                    i,
                    data_field,
                    ns_field,
                    flat_indices=flat_indices,
                )

                if not TQDM_AVAILABLE and (i + 1) % 10 == 0:
                    logger.info(f"  Completed {i+1}/{n_realizations}")

    total_elapsed = time.time() - start_time
    logger.info("-" * 60)
    logger.info(f"SGS Complete: {n_realizations} realizations in {total_elapsed/60:.1f} minutes")
    logger.info(f"Average time per realization: {total_elapsed/n_realizations:.1f} seconds")
    logger.info(f"Output shape: {full_reals.shape}")
    if checkpoint_paths is not None:
        _write_checkpoint_state(checkpoint_paths, checkpoint_shape, completed, status="completed")

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
        'realizations_ns': full_reals_ns,
        'x': x_ax, 'y': y_ax, 'z': z_ax,
        'model': vario_model,
        'grid_def': grid_def,
        'grid_mask': grid_mask,
        'timing': {'total_seconds': total_elapsed, 'avg_per_real': total_elapsed/n_realizations}
    }


def run(data_path=None, data_dir='data', config=None, output_dir='outputs/grids', grid_mask=None):
    """Run SGS simulation."""
    if not GSTOOLS_AVAILABLE:
        raise RuntimeError("gstools is required for SGS")

    if data_path:
        df = pd.read_csv(data_path)
    else:
        from .normal_score import run as run_nst
        _, df = run_nst(data_dir=data_dir, config=config)

    grade_field = (config or {}).get('grade_field', 'tgc_pct')
    validate_conditioning_data(df, grade_field=grade_field)

    from .variography import run as run_variography
    vario_model, _, _ = run_variography(
        data_path=data_path,
        data_dir=data_dir,
        config=config,
        output_dir=os.path.join(os.path.dirname(output_dir), "figures"),
    )

    from .normal_score import NormalScoreTransform
    nst = NormalScoreTransform()

    if 'tgc_ns' not in df.columns:
        weights = df['decluster_weight'].values if 'decluster_weight' in df.columns else None
        nst.fit(df[grade_field].values, weights)
    else:
        nst.fit(
            df[grade_field].values,
            df['decluster_weight'].values if 'decluster_weight' in df.columns else None
        )

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
    requested_n_jobs = sim_config.get('n_jobs', -1)
    n_jobs = requested_n_jobs
    if n_jobs == -1 and JOBLIB_AVAILABLE:
        import multiprocessing
        n_jobs = min(multiprocessing.cpu_count(), n_real)

    # Reduce automatic parallelism for large grids to limit memory pressure,
    # but honor an explicit user/job configuration such as n_jobs: 7.
    n_cells = grid_def['nx'] * grid_def['ny'] * grid_def['nz']
    if n_cells >= 200_000 and requested_n_jobs == -1:
        n_jobs = min(n_jobs, 4)

    # Chunk kriging calls to avoid massive allocations
    chunk_size = sim_config.get('krige_chunk_size', 5000)
    if chunk_size is not None and chunk_size <= 0:
        chunk_size = None

    search_radius = sim_config.get('search_radius_m')
    max_neighbors = sim_config.get('max_neighbors')
    min_neighbors = sim_config.get('min_neighbors')
    update_every = int(sim_config.get('local_update_every', 1))
    checkpoint_every = int(sim_config.get('checkpoint_every', 5000))
    if update_every < 1:
        raise ValueError("simulation.local_update_every must be >= 1")
    if checkpoint_every < 1:
        raise ValueError("simulation.checkpoint_every must be >= 1")

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
        update_every=update_every,
        config=config,
        output_dir=output_dir,
        grid_mask=grid_mask,
        require_full_neighborhood=bool(sim_config.get('require_full_neighborhood', False)),
        checkpoint_every=checkpoint_every,
    )

    os.makedirs(output_dir, exist_ok=True)
    from .utils.io import save_grid
    save_grid(result, output_dir, prefix='sgs')
    if 'realizations_ns' in result:
        np.save(os.path.join(output_dir, 'sgs_reals_ns.npy'), result['realizations_ns'])
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
