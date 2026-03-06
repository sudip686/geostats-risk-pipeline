"""
Compute a practical variogram reproduction check for SGS realizations.

Outputs:
- outputs/tables/variogram_reproduction_lag.csv
- outputs/tables/variogram_reproduction_summary.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import load_config
from src.variography import estimate_directional_variogram


def _build_grid_points(meta: dict):
    nx, ny, nz = int(meta["nx"]), int(meta["ny"]), int(meta["nz"])
    x = meta["x_min"] + np.arange(nx) * float(meta["dx"])
    y = meta["y_min"] + np.arange(ny) * float(meta["dy"])
    z = meta["z_min"] + np.arange(nz) * float(meta["dz"])
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    return np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])


def run(
    config_path="config/project.yaml",
    outputs_dir="outputs",
    n_real_eval=20,
    max_grid_samples=2500,
):
    cfg = load_config(config_path)
    outputs = Path(outputs_dir)
    tables_dir = outputs / "tables"
    grids_dir = outputs / "grids"
    tables_dir.mkdir(parents=True, exist_ok=True)

    domain = pd.read_csv(outputs / "domain_data.csv")
    reals = np.load(grids_dir / "sgs_reals.npy")
    meta = json.loads((grids_dir / "sgs_meta.json").read_text(encoding="utf-8"))

    vario_cfg = cfg.get("variogram", {})
    dirs = vario_cfg.get("directions", [])
    n_lags = int(vario_cfg.get("n_lags", 10))
    max_dist = float(vario_cfg.get("max_distance_m", 500))
    max_pairs = int(vario_cfg.get("max_pairs", 100000))

    # Reviewer-facing check: strike + down-dip where pair support is defensible.
    keep_names = {"along_strike", "down_dip"}
    dirs = [d for d in dirs if d.get("name") in keep_names]

    # Target from conditioning data in grade space.
    coords_data = (domain["x"].values, domain["y"].values, domain["z"].values)
    vals_data = domain["tgc_pct"].values

    grid_points = _build_grid_points(meta)
    n_cells = grid_points.shape[0]
    rng = np.random.default_rng(1337)

    # Evaluate a subset of realizations for speed/stability.
    n_real_eval = max(1, min(int(n_real_eval), reals.shape[0]))
    real_idx = np.linspace(0, reals.shape[0] - 1, n_real_eval, dtype=int)

    # Fixed node sample for comparability across realizations.
    if max_grid_samples < n_cells:
        sample_idx = rng.choice(n_cells, size=max_grid_samples, replace=False)
    else:
        sample_idx = np.arange(n_cells)
    pts = grid_points[sample_idx]
    px, py, pz = pts[:, 0], pts[:, 1], pts[:, 2]

    lag_rows = []
    summary = []

    for d in dirs:
        name = d.get("name", "unknown")
        az = float(d.get("azimuth", 0))
        dip = float(d.get("dip", 0))
        tol = float(d.get("tolerance", 22.5))

        bins_t, gamma_t, counts_t = estimate_directional_variogram(
            coords_data,
            vals_data,
            azimuth=az,
            dip=dip,
            tolerance=tol,
            n_lags=n_lags,
            max_dist=max_dist,
            max_pairs=max_pairs,
            dip_positive_down=True,
        )

        gamma_sim_stack = []
        counts_sim_stack = []
        for i in real_idx:
            vals = reals[i].ravel()[sample_idx]
            bins_s, gamma_s, counts_s = estimate_directional_variogram(
                (px, py, pz),
                vals,
                azimuth=az,
                dip=dip,
                tolerance=tol,
                n_lags=n_lags,
                max_dist=max_dist,
                max_pairs=max_pairs,
                dip_positive_down=True,
            )
            if len(gamma_s) == n_lags:
                gamma_sim_stack.append(gamma_s)
                counts_sim_stack.append(counts_s)

        if not gamma_sim_stack or len(gamma_t) != n_lags:
            continue

        gamma_sim_mean = np.nanmean(np.vstack(gamma_sim_stack), axis=0)
        counts_sim_mean = np.nanmean(np.vstack(counts_sim_stack), axis=0)

        valid = (
            np.isfinite(gamma_t)
            & np.isfinite(gamma_sim_mean)
            & (counts_t > 0)
            & (counts_sim_mean > 0)
        )
        if valid.any():
            rmse = float(np.sqrt(np.mean((gamma_sim_mean[valid] - gamma_t[valid]) ** 2)))
            mae = float(np.mean(np.abs(gamma_sim_mean[valid] - gamma_t[valid])))
            corr = float(np.corrcoef(gamma_t[valid], gamma_sim_mean[valid])[0, 1]) if valid.sum() > 1 else float("nan")
        else:
            rmse, mae, corr = float("nan"), float("nan"), float("nan")

        summary.append(
            {
                "direction": name,
                "n_lags_valid": int(valid.sum()),
                "gamma_rmse": rmse,
                "gamma_mae": mae,
                "gamma_corr": corr,
            }
        )

        for lag in range(n_lags):
            lag_rows.append(
                {
                    "direction": name,
                    "lag": lag + 1,
                    "distance_m": float(bins_t[lag]),
                    "target_gamma": float(gamma_t[lag]) if np.isfinite(gamma_t[lag]) else np.nan,
                    "sim_gamma_mean": float(gamma_sim_mean[lag]) if np.isfinite(gamma_sim_mean[lag]) else np.nan,
                    "abs_diff": float(abs(gamma_sim_mean[lag] - gamma_t[lag])) if (np.isfinite(gamma_t[lag]) and np.isfinite(gamma_sim_mean[lag])) else np.nan,
                    "target_pairs": int(counts_t[lag]),
                    "sim_pairs_mean": float(counts_sim_mean[lag]),
                }
            )

    lag_df = pd.DataFrame(lag_rows)
    lag_df.to_csv(tables_dir / "variogram_reproduction_lag.csv", index=False)
    (tables_dir / "variogram_reproduction_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("Saved variogram reproduction outputs.")


if __name__ == "__main__":
    run()
