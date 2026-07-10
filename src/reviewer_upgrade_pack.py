"""Reviewer-facing post-run diagnostics for the canonical graphite workflow."""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .contact_analysis import run as run_contact_analysis
from .confidence_gradient import run as run_confidence_gradient

logger = logging.getLogger(__name__)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _grid_axes_from_config(config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid = (config or {}).get("grid", {}) or {}
    origin = grid.get("origin_xyz")
    if not origin or len(origin) != 3:
        raise FileNotFoundError("Grid origin is missing from config; cannot build reviewer-upgrade maps.")
    x0, y0, z0 = [float(v) for v in origin]
    nx = int(grid.get("nx", 0))
    ny = int(grid.get("ny", 0))
    nz = int(grid.get("nz", 0))
    dx = float(grid.get("dx", 0))
    dy = float(grid.get("dy", 0))
    dz = float(grid.get("dz", 0))
    if min(nx, ny, nz) <= 0 or min(dx, dy, dz) <= 0:
        raise FileNotFoundError("Grid dimensions are incomplete in config; cannot build reviewer-upgrade maps.")
    return (
        x0 + np.arange(nx, dtype=float) * dx,
        y0 + np.arange(ny, dtype=float) * dy,
        z0 + np.arange(nz, dtype=float) * dz,
    )


def _save_plan_map(
    values: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    out_path: Path,
    title: str,
    color_label: str,
    cmap: str = "viridis",
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    im = ax.imshow(
        values.T,
        origin="lower",
        aspect="auto",
        extent=[float(x.min()), float(x.max()), float(y.min()), float(y.max())],
        cmap=cmap,
    )
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(color_label)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def _collapse_over_z(values: np.ndarray, graphitic_probability: np.ndarray) -> np.ndarray:
    mask = graphitic_probability >= 0.10
    masked = np.where(mask, values, np.nan)
    collapsed = np.nanmean(masked, axis=2)
    if np.isnan(collapsed).all():
        return np.nanmean(values, axis=2)
    fallback = np.nanmean(values, axis=2)
    return np.where(np.isfinite(collapsed), collapsed, fallback)


def _run_vertical_continuity(output_dir: Path) -> dict:
    tables_dir = output_dir / "tables"
    support_specs = [
        ("reporting_support", tables_dir / "validation_metrics.json"),
        ("simulation_support", tables_dir / "validation_metrics_2m.json"),
    ]
    rows: list[dict[str, float | str]] = []
    selected: dict[str, dict] = {}
    for support_name, path in support_specs:
        if not path.exists():
            continue
        metrics = _load_json(path)
        rows.append(
            {
                "support_name": support_name,
                "support_dx_m": float(metrics.get("support_dx", np.nan)),
                "support_dy_m": float(metrics.get("support_dy", np.nan)),
                "support_dz_m": float(metrics.get("support_dz", np.nan)),
                "swath_corr_x": float(metrics.get("swath_corr_x", np.nan)),
                "swath_corr_y": float(metrics.get("swath_corr_y", np.nan)),
                "swath_corr_z": float(metrics.get("swath_corr_z", np.nan)),
                "swath_coverage_pct": float(metrics.get("swath_coverage_pct", np.nan)),
                "hist_overlap": float(metrics.get("hist_overlap", np.nan)),
                "qq_rmse": float(metrics.get("qq_rmse", np.nan)),
            }
        )
        selected[support_name] = metrics

    if not rows:
        raise FileNotFoundError("Validation metrics are missing; vertical continuity cannot be summarized yet.")

    ladder_path = tables_dir / "support_ladder_summary.csv"
    pd.DataFrame(rows).to_csv(ladder_path, index=False)

    reference = selected.get("reporting_support") or next(iter(selected.values()))
    plane_mean = float(np.nanmean([reference.get("swath_corr_x", np.nan), reference.get("swath_corr_y", np.nan)]))
    z_corr = float(reference.get("swath_corr_z", np.nan))
    summary = {
        "preferred_support": "reporting_support" if "reporting_support" in selected else "simulation_support",
        "plane_mean_swath_corr": plane_mean,
        "normal_direction_swath_corr": z_corr,
        "normal_direction_gap": float(plane_mean - z_corr) if np.isfinite(plane_mean) and np.isfinite(z_corr) else np.nan,
        "normal_to_plane_ratio": float(z_corr / plane_mean) if abs(plane_mean) > 1e-9 and np.isfinite(z_corr) else np.nan,
        "interpretation": (
            "Thickness-normal continuity remains the weakest resolved direction in the current reviewer pack."
            if np.isfinite(z_corr) and np.isfinite(plane_mean) and z_corr < plane_mean
            else "Normal-direction continuity is not weaker than the in-plane mean in the available validation metrics."
        ),
        "available_supports": list(selected.keys()),
    }
    summary_path = tables_dir / "vertical_continuity_summary.json"
    _write_json(summary_path, summary)
    return {
        "support_ladder_summary": str(ladder_path),
        "vertical_continuity_summary": str(summary_path),
    }


def _load_probability_stack(output_dir: Path) -> tuple[list[str], np.ndarray]:
    prob_dir = output_dir / "domains" / "categorical"
    codes_path = prob_dir / "domain_codes.json"
    if not codes_path.exists():
        raise FileNotFoundError("Domain probability codes are missing; domain uncertainty cannot be summarized yet.")
    codes = _load_json(codes_path)
    categories = list(codes.get("categories", []))
    if not categories:
        raise FileNotFoundError("Domain category list is empty; domain uncertainty cannot be summarized yet.")
    arrays = []
    for name in categories:
        path = prob_dir / f"domain_probability_{name}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing probability grid for domain '{name}'.")
        arrays.append(np.load(path).astype(np.float32))
    stack = np.stack(arrays, axis=0)
    totals = stack.sum(axis=0, keepdims=True)
    return categories, np.divide(stack, totals, out=np.zeros_like(stack), where=totals > 0)


def _run_domain_uncertainty(output_dir: Path, config: dict, top_n: int) -> dict:
    grids_dir = output_dir / "grids"
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    categories, prob_stack = _load_probability_stack(output_dir)
    x, y, z = _grid_axes_from_config(config)

    safe_prob = np.clip(prob_stack, 1e-12, 1.0)
    entropy = -(safe_prob * np.log(safe_prob)).sum(axis=0) / math.log(len(categories))
    max_prob = prob_stack.max(axis=0)
    graphitic_indices = [idx for idx, name in enumerate(categories) if name in {"fresh_graphitic", "weathered_graphitic"}]
    if not graphitic_indices:
        raise FileNotFoundError("Graphitic domain categories are missing; domain uncertainty cannot be summarized yet.")
    graphitic_probability = prob_stack[graphitic_indices].sum(axis=0)

    np.save(grids_dir / "domain_entropy.npy", entropy.astype(np.float32))
    np.save(grids_dir / "domain_max_probability.npy", max_prob.astype(np.float32))
    np.save(grids_dir / "graphitic_domain_probability.npy", graphitic_probability.astype(np.float32))

    entropy_plan = _collapse_over_z(entropy, graphitic_probability)
    stability_plan = _collapse_over_z(max_prob, graphitic_probability)
    _save_plan_map(
        entropy_plan,
        x,
        y,
        figures_dir / "domain_entropy_map.png",
        "Domain Uncertainty (graphitic-weighted entropy)",
        "Normalized Entropy",
        cmap="magma",
    )
    _save_plan_map(
        stability_plan,
        x,
        y,
        figures_dir / "domain_stability_map.png",
        "Domain Stability (graphitic-weighted max probability)",
        "Max Domain Probability",
        cmap="viridis",
    )

    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    hotspot_df = pd.DataFrame(
        {
            "x": xx.ravel(),
            "y": yy.ravel(),
            "z": zz.ravel(),
            "domain_entropy": entropy.ravel(),
            "domain_max_probability": max_prob.ravel(),
            "graphitic_probability": graphitic_probability.ravel(),
        }
    )
    hotspot_df = hotspot_df[hotspot_df["graphitic_probability"] >= 0.10].dropna()
    hotspots = hotspot_df.sort_values(["domain_entropy", "graphitic_probability"], ascending=[False, False]).head(top_n)
    hotspot_path = tables_dir / "domain_uncertainty_hotspots.csv"
    hotspots.to_csv(hotspot_path, index=False)

    summary = {
        "categories": categories,
        "mean_domain_entropy": float(np.nanmean(entropy)),
        "p90_domain_entropy": float(np.nanpercentile(entropy, 90)),
        "mean_graphitic_probability": float(np.nanmean(graphitic_probability)),
        "cells_entropy_ge_0_60_pct": float(np.nanmean(entropy >= 0.60) * 100.0),
        "cells_max_probability_lt_0_70_pct": float(np.nanmean(max_prob < 0.70) * 100.0),
        "hotspot_count": int(len(hotspots)),
    }
    summary_path = tables_dir / "domain_uncertainty_summary.json"
    _write_json(summary_path, summary)
    return {
        "domain_uncertainty_summary": str(summary_path),
        "domain_uncertainty_hotspots": str(hotspot_path),
        "domain_entropy_map": str(figures_dir / "domain_entropy_map.png"),
        "domain_stability_map": str(figures_dir / "domain_stability_map.png"),
    }


def _run_thickness_geometry(output_dir: Path, config: dict, top_n: int) -> dict:
    grids_dir = output_dir / "grids"
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    prob_dir = output_dir / "domains" / "categorical"
    state_path = prob_dir / "domain_realizations.npy"
    codes_path = prob_dir / "domain_codes.json"
    if not state_path.exists() or not codes_path.exists():
        raise FileNotFoundError("Categorical domain realizations are missing; thickness/geometry risk cannot be summarized yet.")

    codes = _load_json(codes_path)
    cat_to_id = dict(codes.get("cat_to_id", {}))
    graphitic_ids = [cat_to_id[name] for name in ("fresh_graphitic", "weathered_graphitic") if name in cat_to_id]
    if not graphitic_ids:
        raise FileNotFoundError("Graphitic category ids are missing; thickness/geometry risk cannot be summarized yet.")

    reals = np.load(state_path, mmap_mode="r")
    dz = float(((config or {}).get("grid", {}) or {}).get("dz", 0))
    if dz <= 0:
        raise FileNotFoundError("Grid dz is missing from config; thickness/geometry risk cannot be summarized yet.")

    graphitic_mask = np.isin(reals, np.asarray(graphitic_ids, dtype=np.int16))
    thickness = graphitic_mask.sum(axis=3, dtype=np.int32).astype(np.float32) * dz
    p10, p50, p90 = np.percentile(thickness, [10, 50, 90], axis=0)
    presence_probability = np.mean(thickness > 0.0, axis=0, dtype=np.float32)
    aperture = np.divide(
        (p90 - p10),
        p50,
        out=np.full_like(p50, np.nan, dtype=np.float32),
        where=np.abs(p50) > 1e-9,
    ) * 100.0

    np.save(grids_dir / "graphitic_thickness_p10.npy", p10.astype(np.float32))
    np.save(grids_dir / "graphitic_thickness_p50.npy", p50.astype(np.float32))
    np.save(grids_dir / "graphitic_thickness_p90.npy", p90.astype(np.float32))
    np.save(grids_dir / "graphitic_thickness_presence_probability.npy", presence_probability.astype(np.float32))
    np.save(grids_dir / "graphitic_thickness_aperture_pct.npy", aperture.astype(np.float32))

    x, y, _z = _grid_axes_from_config(config)
    _save_plan_map(
        p50,
        x,
        y,
        figures_dir / "graphitic_thickness_p50_map.png",
        "Median Graphitic Thickness",
        "Thickness (m)",
        cmap="cividis",
    )
    _save_plan_map(
        aperture,
        x,
        y,
        figures_dir / "graphitic_thickness_aperture_map.png",
        "Thickness/Geometry Uncertainty",
        "Relative Aperture (%)",
        cmap="inferno",
    )

    xx, yy = np.meshgrid(x, y, indexing="ij")
    hotspot_df = pd.DataFrame(
        {
            "x": xx.ravel(),
            "y": yy.ravel(),
            "presence_probability": presence_probability.ravel(),
            "thickness_p10_m": p10.ravel(),
            "thickness_p50_m": p50.ravel(),
            "thickness_p90_m": p90.ravel(),
            "thickness_aperture_pct": aperture.ravel(),
        }
    )
    hotspot_df = hotspot_df[(hotspot_df["presence_probability"] >= 0.10) | (hotspot_df["thickness_p50_m"] > 0.0)].dropna()
    hotspots = hotspot_df.sort_values(["thickness_aperture_pct", "thickness_p50_m"], ascending=[False, False]).head(top_n)
    hotspot_path = tables_dir / "thickness_geometry_hotspots.csv"
    hotspots.to_csv(hotspot_path, index=False)

    summary = {
        "mean_p50_graphitic_thickness_m": float(np.nanmean(p50)),
        "median_p50_graphitic_thickness_m": float(np.nanmedian(p50)),
        "mean_presence_probability": float(np.nanmean(presence_probability)),
        "cells_aperture_ge_100pct": float(np.nanmean(aperture >= 100.0) * 100.0),
        "cells_presence_ge_0_80_pct": float(np.nanmean(presence_probability >= 0.80) * 100.0),
        "hotspot_count": int(len(hotspots)),
        "interpretation": (
            "This geometry/thickness pack reflects graphitic-domain occupancy uncertainty under the current structural frame; "
            "it does not replace a future local-anisotropy or unfolding model."
        ),
    }
    summary_path = tables_dir / "thickness_geometry_summary.json"
    _write_json(summary_path, summary)
    return {
        "thickness_geometry_summary": str(summary_path),
        "thickness_geometry_hotspots": str(hotspot_path),
        "graphitic_thickness_p50_map": str(figures_dir / "graphitic_thickness_p50_map.png"),
        "graphitic_thickness_aperture_map": str(figures_dir / "graphitic_thickness_aperture_map.png"),
    }


def run(output_dir: str = "outputs", config: dict | None = None, top_n: int = 50) -> dict:
    output_path = Path(output_dir)
    tables_dir = output_path / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    pack_cfg = ((config or {}).get("postrun_review_pack", {}) or {})
    if not pack_cfg.get("enabled", True):
        payload = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "status": "disabled",
            "tasks": {},
        }
        _write_json(tables_dir / "postrun_review_pack_status.json", payload)
        return payload

    fail_on_error = bool(pack_cfg.get("fail_on_error", False))
    enabled_tasks = dict((pack_cfg.get("tasks", {}) or {}))
    task_order = [
        ("vertical_continuity", lambda: _run_vertical_continuity(output_path)),
        ("contact_analysis", lambda: run_contact_analysis(output_dir=str(output_path))),
        ("domain_uncertainty", lambda: _run_domain_uncertainty(output_path, config or {}, top_n)),
        ("thickness_geometry", lambda: _run_thickness_geometry(output_path, config or {}, top_n)),
        ("confidence_gradient", lambda: run_confidence_gradient(output_dir=str(output_path), top_n=top_n)),
    ]

    results: dict[str, dict] = {}
    for task_name, runner in task_order:
        if task_name in enabled_tasks and not bool(enabled_tasks[task_name]):
            results[task_name] = {"status": "disabled"}
            continue
        try:
            task_result = runner()
            results[task_name] = {"status": "completed", "outputs": task_result}
        except FileNotFoundError as exc:
            results[task_name] = {"status": "skipped", "reason": str(exc)}
        except Exception as exc:  # pragma: no cover - defensive reporting
            logger.exception("Post-run reviewer task '%s' failed", task_name)
            if fail_on_error:
                raise
            results[task_name] = {"status": "failed", "error": str(exc)}

    status = "completed"
    if any(task.get("status") == "failed" for task in results.values()):
        status = "completed_with_failures"
    elif any(task.get("status") == "skipped" for task in results.values()):
        status = "completed_with_skips"

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "tasks": results,
        "deferred_high_effort_gap": "local anisotropy / structural unfolding",
    }
    _write_json(tables_dir / "postrun_review_pack_status.json", payload)
    return payload

