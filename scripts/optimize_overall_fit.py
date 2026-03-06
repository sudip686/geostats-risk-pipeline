"""
Optimize SGS validation fit by sweeping key configuration parameters.

Strategy:
1) Screening runs with reduced realizations for speed.
2) Confirm top candidates with full realizations.
3) Persist ranked results and best-fit config.
"""

from __future__ import annotations

import copy
import itertools
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "project.yaml"
TMP_CONFIG_DIR = ROOT / "config" / "_fit_tuning"
OUTPUT_ROOT = ROOT / "outputs_fit_tuning"


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def run_pipeline(config_path: Path, output_dir: Path) -> bool:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "src.run_all",
        "--config",
        str(config_path),
        "--output",
        str(output_dir),
    ]
    p = subprocess.run(cmd, cwd=ROOT)
    return p.returncode == 0


def score_from_metrics(metrics: dict) -> float:
    # Higher is better.
    # Blend distributional fit and spatial continuity.
    hist = float(metrics.get("hist_overlap", 0.0))
    qq_rmse = float(metrics.get("qq_rmse", 10.0))
    swx = float(metrics.get("swath_corr_x", 0.0))
    swy = float(metrics.get("swath_corr_y", 0.0))
    swz = float(metrics.get("swath_corr_z", 0.0))
    mean_data = float(metrics.get("mean_data", 0.0))
    mean_sim = float(metrics.get("mean_sim", 0.0))
    std_data = float(metrics.get("std_data", 1.0))
    std_sim = float(metrics.get("std_sim", 0.0))

    qq_term = 1.0 / (1.0 + qq_rmse)
    mean_term = 1.0 / (1.0 + abs(mean_sim - mean_data))
    std_term = 1.0 / (1.0 + abs(std_sim - std_data))

    return (
        0.30 * hist
        + 0.20 * qq_term
        + 0.15 * swx
        + 0.15 * swy
        + 0.10 * swz
        + 0.05 * mean_term
        + 0.05 * std_term
    )


def build_candidates(base_cfg: dict) -> list[dict]:
    normal_ranges = [60, 80, 100, 120, 150]
    nugget_ratios = [0.20, 0.25, 0.30]
    search_normals = [60, 80, 100]
    neighbor_pairs = [(6, 20), (8, 24), (10, 30)]

    candidates = []
    for nr, ng, srn, (mn, mx) in itertools.product(
        normal_ranges, nugget_ratios, search_normals, neighbor_pairs
    ):
        cfg = copy.deepcopy(base_cfg)
        cfg.setdefault("variogram", {}).setdefault("anisotropy", {}).setdefault(
            "ranges_m", {}
        )["normal"] = nr
        cfg.setdefault("variogram", {}).setdefault("tuning", {})["enabled"] = True
        cfg["variogram"]["tuning"]["nugget_ratio"] = ng
        cfg.setdefault("simulation", {})["search_radius_m"] = [250, 120, srn]
        cfg["simulation"]["min_neighbors"] = mn
        cfg["simulation"]["max_neighbors"] = mx
        cfg.setdefault("sensitivity", {})["enabled"] = False
        candidates.append(cfg)

    # De-duplicate by key tuple
    seen = set()
    unique = []
    for c in candidates:
        key = (
            c["variogram"]["anisotropy"]["ranges_m"]["normal"],
            c["variogram"]["tuning"]["nugget_ratio"],
            tuple(c["simulation"]["search_radius_m"]),
            c["simulation"]["min_neighbors"],
            c["simulation"]["max_neighbors"],
        )
        if key not in seen:
            seen.add(key)
            unique.append(c)

    # Keep a focused subset around practical values.
    # Deterministic pick to keep runtime bounded.
    picks = []
    target_keys = [
        (100, 0.25, (250, 120, 80), 8, 24),
        (120, 0.25, (250, 120, 80), 8, 24),
        (150, 0.25, (250, 120, 80), 8, 24),
        (80, 0.25, (250, 120, 80), 8, 24),
        (100, 0.20, (250, 120, 80), 8, 24),
        (100, 0.30, (250, 120, 80), 8, 24),
        (100, 0.25, (250, 120, 60), 8, 24),
        (100, 0.25, (250, 120, 100), 8, 24),
        (120, 0.30, (250, 120, 80), 8, 24),
        (80, 0.20, (250, 120, 80), 8, 24),
    ]
    for c in unique:
        key = (
            c["variogram"]["anisotropy"]["ranges_m"]["normal"],
            c["variogram"]["tuning"]["nugget_ratio"],
            tuple(c["simulation"]["search_radius_m"]),
            c["simulation"]["min_neighbors"],
            c["simulation"]["max_neighbors"],
        )
        if key in target_keys:
            picks.append(c)
    return picks


def main():
    base_cfg = load_yaml(CONFIG_PATH)
    candidates = build_candidates(base_cfg)

    TMP_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    screen_rows = []
    print(f"Screening {len(candidates)} candidates...")
    for i, cfg in enumerate(candidates, start=1):
        cfg = copy.deepcopy(cfg)
        cfg["simulation"]["n_real"] = 40  # screening speed
        cfg.setdefault("ci", {})["n_real"] = 20

        tag = f"screen_{i:02d}"
        cfg_path = TMP_CONFIG_DIR / f"{tag}.yaml"
        out_dir = OUTPUT_ROOT / tag
        save_yaml(cfg_path, cfg)

        ok = run_pipeline(cfg_path, out_dir)
        row = {
            "tag": tag,
            "ok": ok,
            "normal_range": cfg["variogram"]["anisotropy"]["ranges_m"]["normal"],
            "nugget_ratio": cfg["variogram"]["tuning"]["nugget_ratio"],
            "search_radius": cfg["simulation"]["search_radius_m"],
            "min_neighbors": cfg["simulation"]["min_neighbors"],
            "max_neighbors": cfg["simulation"]["max_neighbors"],
        }
        if ok:
            metrics_path = out_dir / "tables" / "validation_metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            row.update(metrics)
            row["score"] = score_from_metrics(metrics)
        else:
            row["score"] = -1.0
        screen_rows.append(row)
        print(f"{tag}: ok={ok}, score={row['score']:.4f}")

    screen_df = pd.DataFrame(screen_rows).sort_values("score", ascending=False)
    screen_df.to_csv(OUTPUT_ROOT / "screen_results.csv", index=False)

    top = screen_df[screen_df["ok"]].head(2).copy()
    if top.empty:
        raise RuntimeError("No successful screening run found.")

    confirm_rows = []
    print("Confirming top candidates with n_real=200...")
    for rank, (_, srow) in enumerate(top.iterrows(), start=1):
        tag = srow["tag"]
        cfg = load_yaml(TMP_CONFIG_DIR / f"{tag}.yaml")
        cfg["simulation"]["n_real"] = 200
        cfg_path = TMP_CONFIG_DIR / f"confirm_{rank:02d}_{tag}.yaml"
        out_dir = OUTPUT_ROOT / f"confirm_{rank:02d}_{tag}"
        save_yaml(cfg_path, cfg)

        ok = run_pipeline(cfg_path, out_dir)
        row = {
            "tag": f"confirm_{rank:02d}_{tag}",
            "ok": ok,
            "normal_range": cfg["variogram"]["anisotropy"]["ranges_m"]["normal"],
            "nugget_ratio": cfg["variogram"]["tuning"]["nugget_ratio"],
            "search_radius": cfg["simulation"]["search_radius_m"],
            "min_neighbors": cfg["simulation"]["min_neighbors"],
            "max_neighbors": cfg["simulation"]["max_neighbors"],
        }
        if ok:
            metrics = json.loads(
                (out_dir / "tables" / "validation_metrics.json").read_text(
                    encoding="utf-8"
                )
            )
            row.update(metrics)
            row["score"] = score_from_metrics(metrics)
        else:
            row["score"] = -1.0
        confirm_rows.append(row)
        print(f"{row['tag']}: ok={ok}, score={row['score']:.4f}")

    confirm_df = pd.DataFrame(confirm_rows).sort_values("score", ascending=False)
    confirm_df.to_csv(OUTPUT_ROOT / "confirm_results.csv", index=False)

    best = confirm_df.iloc[0].to_dict()
    best_cfg_tag = best["tag"].replace("confirm_01_", "").replace("confirm_02_", "")
    source_cfg = None
    # Find matching confirm config by exact tag
    for p in TMP_CONFIG_DIR.glob("confirm_*_screen_*.yaml"):
        if p.stem == best["tag"]:
            source_cfg = p
            break
    if source_cfg is None:
        raise RuntimeError("Best config file could not be located.")

    best_cfg = load_yaml(source_cfg)
    best_cfg_path = ROOT / "config" / "project_best_fit.yaml"
    save_yaml(best_cfg_path, best_cfg)

    # Also apply best to active project config.
    save_yaml(CONFIG_PATH, best_cfg)

    summary = {
        "best_tag": best["tag"],
        "best_score": float(best["score"]),
        "best_config_path": str(best_cfg_path),
        "applied_to_project_yaml": True,
        "screen_results_csv": str(OUTPUT_ROOT / "screen_results.csv"),
        "confirm_results_csv": str(OUTPUT_ROOT / "confirm_results.csv"),
    }
    (OUTPUT_ROOT / "best_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("Best configuration applied to config/project.yaml")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
