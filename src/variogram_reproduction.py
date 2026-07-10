from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import load_config
from src.variography import estimate_directional_variogram

logger = logging.getLogger("variogram_reproduction")


def _grid_points(meta: dict) -> np.ndarray:
    nx = int(meta["nx"])
    ny = int(meta["ny"])
    nz = int(meta["nz"])
    x = float(meta["x_min"]) + np.arange(nx) * float(meta["dx"])
    y = float(meta["y_min"]) + np.arange(ny) * float(meta["dy"])
    z = float(meta["z_min"]) + np.arange(nz) * float(meta["dz"])
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])


def _default_support_min_distance(grid_meta: dict) -> float:
    dx = float(grid_meta.get("dx", 0.0))
    dy = float(grid_meta.get("dy", 0.0))
    dz = float(grid_meta.get("dz", 0.0))
    # Block-support guardrail: ignore shortest lags where point-vs-block mismatch dominates.
    return 0.75 * float(np.sqrt(dx * dx + dy * dy + dz * dz))


def _direction_summary(
    df: pd.DataFrame,
    direction: str,
    min_pairs: int,
    min_distance_m: float | None,
) -> dict:
    dsub = df[(df["space"] == "nst") & (df["direction"] == direction)].copy()
    total_lags = int(len(dsub))
    if total_lags == 0:
        return {
            "rmse": np.nan,
            "mae": np.nan,
            "bias": np.nan,
            "usable_lags": 0,
            "total_lags": 0,
            "missing_lags": 0,
            "lag_coverage_pct": 0.0,
        }

    mask = (
        dsub["gamma_target"].notna()
        & dsub["gamma_sim_mean"].notna()
        & (dsub["target_pairs"] >= min_pairs)
        & (dsub["sim_pairs_mean"] >= float(min_pairs))
    )
    if min_distance_m is not None:
        mask = mask & (dsub["distance_m"] >= float(min_distance_m))
    use = dsub[mask]
    usable_lags = int(len(use))
    missing_lags = int(total_lags - usable_lags)
    coverage = 100.0 * float(usable_lags) / max(1.0, float(total_lags))

    if usable_lags == 0:
        return {
            "rmse": np.nan,
            "mae": np.nan,
            "bias": np.nan,
            "usable_lags": usable_lags,
            "total_lags": total_lags,
            "missing_lags": missing_lags,
            "lag_coverage_pct": coverage,
        }

    diff = use["gamma_sim_mean"].to_numpy(dtype=float) - use["gamma_target"].to_numpy(dtype=float)
    return {
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "mae": float(np.mean(np.abs(diff))),
        "bias": float(np.mean(diff)),
        "usable_lags": usable_lags,
        "total_lags": total_lags,
        "missing_lags": missing_lags,
        "lag_coverage_pct": coverage,
    }


def _build_reproduction_summary(
    df: pd.DataFrame,
    min_pairs: int,
    min_distance_m: float | None,
    required_weighted_rmse: float,
    required_along_rmse: float,
    max_missing_lags_per_dir: int,
) -> dict:
    direction_weights = {"along_strike": 0.45, "down_dip": 0.40, "normal_to_plane": 0.15}
    by_dir = {}
    weighted_rmse = 0.0
    used_weight = 0.0
    total_missing = 0
    for direction, weight in direction_weights.items():
        stat = _direction_summary(df, direction, min_pairs=min_pairs, min_distance_m=min_distance_m)
        by_dir[direction] = stat
        total_missing += int(stat["missing_lags"])
        rmse = float(stat["rmse"])
        if np.isfinite(rmse):
            weighted_rmse += weight * rmse
            used_weight += weight

    weighted = float(weighted_rmse / used_weight) if used_weight > 0 else np.nan
    along_rmse = float(by_dir["along_strike"]["rmse"])
    missing_ok = all(int(by_dir[d]["missing_lags"]) <= int(max_missing_lags_per_dir) for d in by_dir)

    gates = {
        "weighted_rmse_ok": bool(np.isfinite(weighted) and weighted <= required_weighted_rmse),
        "along_strike_rmse_ok": bool(np.isfinite(along_rmse) and along_rmse <= required_along_rmse),
        "missing_lags_ok": bool(missing_ok),
    }
    gates["all_pass"] = bool(all(gates.values()))

    return {
        "space": "nst",
        "min_pairs_for_scoring": int(min_pairs),
        "min_distance_m_for_scoring": None if min_distance_m is None else float(min_distance_m),
        "required_weighted_rmse": float(required_weighted_rmse),
        "required_along_strike_rmse": float(required_along_rmse),
        "max_missing_lags_per_direction": int(max_missing_lags_per_dir),
        "weighted_rmse": weighted,
        "along_strike_rmse": along_rmse,
        "total_missing_lags": int(total_missing),
        "direction_metrics": by_dir,
        "acceptance": gates,
    }


def _compute_for_space(
    label: str,
    dirs: list[dict],
    n_lags: int,
    max_dist: float,
    max_pairs: int,
    coords_target: tuple[np.ndarray, np.ndarray, np.ndarray],
    vals_target: np.ndarray,
    pts_sim: tuple[np.ndarray, np.ndarray, np.ndarray],
    sim_values_by_real: list[np.ndarray],
    sim_real_ids: list[int],
    debug_rows: list[dict],
    lag_rows: list[dict],
    angle_rows: list[dict],
) -> pd.DataFrame:
    t0 = time.time()
    logger.info("Computing %s-space variograms for %d directions", label, len(dirs))
    rows = []
    for d_idx, d in enumerate(dirs, start=1):
        name = d.get("name")
        az = float(d.get("azimuth", 0))
        dip = float(d.get("dip", 0))
        tol = float(d.get("tolerance", 22.5))
        logger.info(
            "[%s] Direction %d/%d: %s (az=%.1f dip=%.1f tol=%.1f)",
            label,
            d_idx,
            len(dirs),
            name,
            az,
            dip,
            tol,
        )

        bins_t, gamma_t, counts_t, dbg_t = estimate_directional_variogram(
            coords_target,
            vals_target,
            azimuth=az,
            dip=dip,
            tolerance=tol,
            n_lags=n_lags,
            max_dist=max_dist,
            max_pairs=max_pairs,
            dip_positive_down=True,
            return_debug=True,
        )
        debug_rows.append(
            {
                "space": label,
                "direction": name,
                "source": "target",
                "realization": -1,
                "azimuth_deg": az,
                "dip_deg": dip,
                "tolerance_deg": tol,
                **dbg_t,
            }
        )
        angle_bins_t = dbg_t.get("angle_hist_bins_deg", [])
        angle_counts_t = dbg_t.get("angle_hist_counts", [])
        angle_total_t = float(sum(angle_counts_t))
        for i, cnt in enumerate(angle_counts_t):
            if i + 1 >= len(angle_bins_t):
                break
            angle_rows.append(
                {
                    "space": label,
                    "direction": name,
                    "source": "target",
                    "realization": -1,
                    "angle_bin_start_deg": float(angle_bins_t[i]),
                    "angle_bin_end_deg": float(angle_bins_t[i + 1]),
                    "count": int(cnt),
                    "pct_of_angle_evaluated": 100.0 * float(cnt) / max(1.0, angle_total_t),
                }
            )
        total_t = float(np.sum(counts_t)) if len(counts_t) else 0.0
        for lag_idx, cnt in enumerate(counts_t, start=1):
            lag_rows.append(
                {
                    "space": label,
                    "direction": name,
                    "source": "target",
                    "realization": -1,
                    "lag": lag_idx,
                    "count": int(cnt),
                    "pct_of_source_pairs": 100.0 * float(cnt) / max(1.0, total_t),
                }
            )

        sim_stack = []
        count_stack = []
        n_reals = len(sim_values_by_real)
        for real_idx, vals in enumerate(sim_values_by_real, start=1):
            _, gamma_s, count_s, dbg_s = estimate_directional_variogram(
                pts_sim,
                vals,
                azimuth=az,
                dip=dip,
                tolerance=tol,
                n_lags=n_lags,
                max_dist=max_dist,
                max_pairs=max_pairs,
                dip_positive_down=True,
                return_debug=True,
            )
            real_id = int(sim_real_ids[real_idx - 1]) if real_idx - 1 < len(sim_real_ids) else real_idx - 1
            debug_rows.append(
                {
                    "space": label,
                    "direction": name,
                    "source": "sim",
                    "realization": real_id,
                    "azimuth_deg": az,
                    "dip_deg": dip,
                    "tolerance_deg": tol,
                    **dbg_s,
                }
            )
            angle_bins_s = dbg_s.get("angle_hist_bins_deg", [])
            angle_counts_s = dbg_s.get("angle_hist_counts", [])
            angle_total_s = float(sum(angle_counts_s))
            for i, cnt in enumerate(angle_counts_s):
                if i + 1 >= len(angle_bins_s):
                    break
                angle_rows.append(
                    {
                        "space": label,
                        "direction": name,
                        "source": "sim",
                        "realization": real_id,
                        "angle_bin_start_deg": float(angle_bins_s[i]),
                        "angle_bin_end_deg": float(angle_bins_s[i + 1]),
                        "count": int(cnt),
                        "pct_of_angle_evaluated": 100.0 * float(cnt) / max(1.0, angle_total_s),
                    }
                )
            total_s = float(np.sum(count_s)) if len(count_s) else 0.0
            for lag_idx, cnt in enumerate(count_s, start=1):
                lag_rows.append(
                    {
                        "space": label,
                        "direction": name,
                        "source": "sim",
                        "realization": real_id,
                        "lag": lag_idx,
                        "count": int(cnt),
                        "pct_of_source_pairs": 100.0 * float(cnt) / max(1.0, total_s),
                    }
                )
            if len(gamma_s) == n_lags:
                sim_stack.append(gamma_s)
                count_stack.append(count_s)
            if real_idx == 1 or real_idx == n_reals or real_idx % 5 == 0:
                logger.info(
                    "[%s:%s] realizations processed %d/%d",
                    label,
                    name,
                    real_idx,
                    n_reals,
                )

        if not sim_stack or len(gamma_t) != n_lags:
            logger.warning("[%s:%s] skipped due to insufficient lag support", label, name)
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            gsim = np.nanmean(np.vstack(sim_stack), axis=0)
            csim = np.nanmean(np.vstack(count_stack), axis=0)

        for lag in range(n_lags):
            rows.append(
                {
                    "space": label,
                    "direction": name,
                    "lag": lag + 1,
                    "distance_m": float(bins_t[lag]),
                    "gamma_target": float(gamma_t[lag]) if np.isfinite(gamma_t[lag]) else np.nan,
                    "gamma_sim_mean": float(gsim[lag]) if np.isfinite(gsim[lag]) else np.nan,
                    "target_pairs": int(counts_t[lag]),
                    "sim_pairs_mean": float(csim[lag]),
                }
            )
        logger.info("[%s:%s] done", label, name)
    logger.info("Completed %s-space variograms in %.1fs", label, time.time() - t0)
    return pd.DataFrame(rows)


def run(
    config_path: str,
    outputs_dir: str,
    n_real_eval: int = 20,
    max_grid_samples: int = 2500,
    min_pairs: int = 200,
    support_aware: bool = False,
    min_distance_m: float | None = None,
    required_weighted_rmse: float = 0.35,
    required_along_strike_rmse: float = 0.50,
    max_missing_lags_per_dir: int = 2,
) -> None:
    t0 = time.time()
    logger.info("Loading config: %s", config_path)
    outputs = Path(outputs_dir)
    cfg = load_config(config_path)
    logger.info("Outputs directory: %s", outputs)

    tables_dir = outputs / "tables"
    figs_dir = outputs / "figures"
    grids_dir = outputs / "grids"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Reading domain and NST data...")
    domain = pd.read_csv(outputs / "domain_data.csv")
    nst_df = pd.read_csv(outputs / "nst_data.csv")
    logger.info("Loaded domain rows=%d, nst rows=%d", len(domain), len(nst_df))

    # Use uncalibrated realizations for structural checks to avoid calibration-space mismatch.
    reals_path = grids_dir / "sgs_reals.npy"
    logger.info("Loading realizations from %s", reals_path)
    reals = np.load(reals_path)
    logger.info("Realizations shape=%s", tuple(reals.shape))

    grid_meta = json.loads((grids_dir / "sgs_meta.json").read_text(encoding="utf-8"))
    points = _grid_points(grid_meta)
    logger.info("Grid points loaded: %d", points.shape[0])
    if support_aware and min_distance_m is None:
        min_distance_m = _default_support_min_distance(grid_meta)
    if support_aware:
        logger.info("Support-aware scoring enabled (min_distance_m=%.2f)", float(min_distance_m))

    vario_cfg = cfg.get("variogram", {})
    dirs = vario_cfg.get("directions", [])
    keep = {"along_strike", "down_dip", "normal_to_plane"}
    dirs = [d for d in dirs if d.get("name") in keep]

    n_lags = int(vario_cfg.get("n_lags", 10))
    max_dist = float(vario_cfg.get("max_distance_m", 500))
    max_pairs = int(vario_cfg.get("max_pairs", 100000))
    logger.info(
        "Variogram settings: directions=%d n_lags=%d max_dist=%.1f max_pairs=%d",
        len(dirs),
        n_lags,
        max_dist,
        max_pairs,
    )

    rng = np.random.default_rng(1337)
    n_cells = points.shape[0]
    sample_idx = rng.choice(n_cells, size=min(max_grid_samples, n_cells), replace=False)
    logger.info("Sampling %d/%d grid cells for diagnostics", sample_idx.shape[0], n_cells)
    pts = points[sample_idx]
    px, py, pz = pts[:, 0], pts[:, 1], pts[:, 2]

    n_real_eval = max(1, min(n_real_eval, reals.shape[0]))
    ridx = np.linspace(0, reals.shape[0] - 1, n_real_eval, dtype=int)
    logger.info("Evaluating %d realizations: first=%d last=%d", n_real_eval, int(ridx[0]), int(ridx[-1]))

    # Grade space (original units): target from domain composites, sim from uncalibrated simulated grades.
    coords_grade = (domain["x"].values, domain["y"].values, domain["z"].values)
    vals_grade = domain["tgc_pct"].values.astype(float)
    sim_grade_by_real = [reals[i].ravel()[sample_idx].astype(float) for i in ridx]

    # NST space: target from nst_data['tgc_ns']; sim from raw SGS normal-score realizations.
    coords_nst = (nst_df["x"].values, nst_df["y"].values, nst_df["z"].values)
    vals_nst = nst_df["tgc_ns"].values.astype(float)

    sim_nst_by_real = []
    reals_ns_path = grids_dir / "sgs_reals_ns.npy"
    if reals_ns_path.exists():
        logger.info("Loading NST realizations from %s", reals_ns_path)
        reals_ns = np.load(reals_ns_path)
        if reals_ns.shape != reals.shape:
            raise ValueError(
                f"Shape mismatch between sgs_reals_ns.npy {tuple(reals_ns.shape)} "
                f"and sgs_reals.npy {tuple(reals.shape)}"
            )
        for i in ridx:
            sim_nst_by_real.append(reals_ns[i].ravel()[sample_idx].astype(float))
    else:
        raise FileNotFoundError(
            f"Missing matched-space NST realizations: {reals_ns_path}. "
            "Regenerate SGS outputs so variogram reproduction uses true NST simulation space."
        )
    logger.info("Built NST simulation vectors: %d", len(sim_nst_by_real))
    debug_rows: list[dict] = []
    lag_rows: list[dict] = []
    angle_rows: list[dict] = []
    ridx_list = [int(v) for v in ridx.tolist()]

    df_grade = _compute_for_space(
        "grade",
        dirs,
        n_lags,
        max_dist,
        max_pairs,
        coords_grade,
        vals_grade,
        (px, py, pz),
        sim_grade_by_real,
        ridx_list,
        debug_rows,
        lag_rows,
        angle_rows,
    )
    df_nst = _compute_for_space(
        "nst",
        dirs,
        n_lags,
        max_dist,
        max_pairs,
        coords_nst,
        vals_nst,
        (px, py, pz),
        sim_nst_by_real,
        ridx_list,
        debug_rows,
        lag_rows,
        angle_rows,
    )

    df = pd.concat([df_nst, df_grade], ignore_index=True)
    out_csv = tables_dir / "variogram_reproduction.csv"
    df.to_csv(out_csv, index=False)
    logger.info("Wrote table: %s (%d rows)", out_csv, len(df))

    # Backward-compatible alias for existing supplement references.
    df.to_csv(tables_dir / "variogram_reproduction_lag.csv", index=False)
    if debug_rows:
        df_debug = pd.DataFrame(debug_rows)
        df_debug["angle_acceptance_pct"] = 100.0 * df_debug["angle_accepted_pairs"] / df_debug["angle_evaluated_pairs"].clip(lower=1)
        df_debug["binned_from_angle_pct"] = 100.0 * df_debug["binned_pairs"] / df_debug["angle_accepted_pairs"].clip(lower=1)
        debug_path = tables_dir / "variogram_reproduction_direction_debug.csv"
        df_debug.to_csv(debug_path, index=False)
        logger.info("Wrote directional debug table: %s (%d rows)", debug_path, len(df_debug))
    if lag_rows:
        lag_occ_path = tables_dir / "variogram_reproduction_lag_occupancy.csv"
        pd.DataFrame(lag_rows).to_csv(lag_occ_path, index=False)
        logger.info("Wrote lag occupancy table: %s (%d rows)", lag_occ_path, len(lag_rows))
    if angle_rows:
        angle_hist_path = tables_dir / "variogram_reproduction_angle_histogram.csv"
        pd.DataFrame(angle_rows).to_csv(angle_hist_path, index=False)
        logger.info("Wrote angle histogram table: %s (%d rows)", angle_hist_path, len(angle_rows))

    summary = _build_reproduction_summary(
        df=df,
        min_pairs=min_pairs,
        min_distance_m=min_distance_m,
        required_weighted_rmse=required_weighted_rmse,
        required_along_rmse=required_along_strike_rmse,
        max_missing_lags_per_dir=max_missing_lags_per_dir,
    )
    summary_path = tables_dir / "variogram_reproduction_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote summary JSON: %s", summary_path)

    # Plot: rows = spaces, cols = directions
    dir_order = ["along_strike", "down_dip", "normal_to_plane"]
    spaces = [s for s in ["nst", "grade"] if s in set(df["space"])]
    fig, axes = plt.subplots(len(spaces), len(dir_order), figsize=(5.2 * len(dir_order), 4.2 * len(spaces)), squeeze=False)

    for r, space in enumerate(spaces):
        for c, direction in enumerate(dir_order):
            ax = axes[r, c]
            dsub = df[(df["space"] == space) & (df["direction"] == direction)].sort_values("lag")
            if dsub.empty:
                ax.axis("off")
                continue
            ax.plot(dsub["distance_m"], dsub["gamma_target"], marker="o", label="Target")
            ax.plot(dsub["distance_m"], dsub["gamma_sim_mean"], marker="s", label="Sim mean")
            title = f"{space.upper()} | {direction}"
            if direction == "normal_to_plane":
                title += " (low pair support caveat)"
            ax.set_title(title)
            ax.set_xlabel("Lag distance (m)")
            ax.set_ylabel("Semivariance")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(figs_dir / "variogram_reproduction.png", dpi=220)
    plt.close(fig)
    logger.info("Saved figure: %s", figs_dir / "variogram_reproduction.png")

    logger.info("Completed variogram reproduction in %.1fs", time.time() - t0)
    print(f"Saved {out_csv} and {figs_dir / 'variogram_reproduction.png'}")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    parser = argparse.ArgumentParser(description="Compute SGS variogram reproduction diagnostics in matched spaces")
    parser.add_argument("--config", default="config/main_config.yaml")
    parser.add_argument("--outputs", default="outputs_fit_tuning/refine/refine_n70_ng020")
    parser.add_argument("--n-real-eval", type=int, default=20)
    parser.add_argument("--max-grid-samples", type=int, default=2500)
    parser.add_argument("--min-pairs", type=int, default=200)
    parser.add_argument("--support-aware", action="store_true")
    parser.add_argument("--min-distance-m", type=float, default=None)
    parser.add_argument("--required-weighted-rmse", type=float, default=0.35)
    parser.add_argument("--required-along-strike-rmse", type=float, default=0.50)
    parser.add_argument("--max-missing-lags-per-dir", type=int, default=2)
    args = parser.parse_args()
    run(
        args.config,
        args.outputs,
        args.n_real_eval,
        args.max_grid_samples,
        args.min_pairs,
        args.support_aware,
        args.min_distance_m,
        args.required_weighted_rmse,
        args.required_along_strike_rmse,
        args.max_missing_lags_per_dir,
    )


if __name__ == "__main__":
    main()
