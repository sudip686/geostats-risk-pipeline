"""
Optimize SGS fit with a variogram-first objective and support-aware diagnostics.

Strategy:
1) Screen candidates with reduced realizations.
2) Confirm top candidates with full realizations.
3) Select best candidate using weighted NST variogram score plus guardrails.
4) Promote winning config as project source-of-truth.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "main_config.yaml"
BEST_CFG_PATH = ROOT / "config" / "main_config.yaml"
TMP_CONFIG_DIR = ROOT / "config" / "_fit_tuning"
OUTPUT_ROOT = ROOT / "outputs_fit_tuning"
BEST_SUMMARY = OUTPUT_ROOT / "best_summary.json"
OPT_MANIFEST = OUTPUT_ROOT / "optimization_manifest.json"

# Runtime-bounded search pools (sampled down by --max-candidates).
TARGET_RANGES = [130, 140, 150, 160, 170]
NUGGET_RATIOS = [0.16, 0.20, 0.24]
NORMAL_RANGES = [60, 70, 80, 90]
SEARCH_RADII = [[240, 120, 70], [250, 120, 70], [260, 130, 80]]
NEIGHBOR_BOUNDS = [(8, 24), (10, 28)]

GUARDRAILS = {
    "hist_overlap_min": 0.97,
    "qq_rmse_max": 0.10,
    "swath_corr_x_min": 0.50,
    "swath_corr_y_min": 0.50,
    "swath_corr_z_min": 0.35,
}


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def enforce_production_invariants(cfg: dict) -> dict:
    """Ensure promoted config always remains publish-safe."""
    out = copy.deepcopy(cfg)
    out.setdefault("simulation", {})["n_real"] = 400
    out.setdefault("ci", {})["n_real"] = 80
    out.setdefault("sensitivity", {})
    if "n_real" in out["sensitivity"]:
        out["sensitivity"]["n_real"] = 40
    return out


def run_command(cmd: list[str]) -> tuple[bool, str]:
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if proc.returncode == 0:
        return True, ""
    stderr = (proc.stderr or "").strip()
    stdout = (proc.stdout or "").strip()
    msg = stderr[-1500:] if stderr else stdout[-1500:]
    return False, msg or f"return_code_{proc.returncode}"


def run_pipeline(config_path: Path, output_dir: Path) -> tuple[bool, str]:
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
    return run_command(cmd)


def run_variogram_reproduction(
    config_path: Path,
    output_dir: Path,
    n_real_eval: int,
    min_pairs: int,
    support_aware: bool,
) -> tuple[bool, str]:
    cmd = [
        sys.executable,
        "src/variogram_reproduction.py",
        "--config",
        str(config_path),
        "--outputs",
        str(output_dir),
        "--n-real-eval",
        str(n_real_eval),
        "--max-grid-samples",
        "2500",
        "--min-pairs",
        str(min_pairs),
    ]
    if support_aware:
        cmd.append("--support-aware")
    return run_command(cmd)


def guardrail_status(metrics: dict) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if float(metrics.get("hist_overlap", 0.0)) < GUARDRAILS["hist_overlap_min"]:
        failures.append("hist_overlap")
    if float(metrics.get("qq_rmse", 999.0)) > GUARDRAILS["qq_rmse_max"]:
        failures.append("qq_rmse")
    if float(metrics.get("swath_corr_x", -1.0)) < GUARDRAILS["swath_corr_x_min"]:
        failures.append("swath_corr_x")
    if float(metrics.get("swath_corr_y", -1.0)) < GUARDRAILS["swath_corr_y_min"]:
        failures.append("swath_corr_y")
    if float(metrics.get("swath_corr_z", -1.0)) < GUARDRAILS["swath_corr_z_min"]:
        failures.append("swath_corr_z")
    return len(failures) == 0, failures


def variogram_components_from_csv(variogram_csv: Path) -> dict:
    if not variogram_csv.exists():
        return {
            "weighted_rmse": math.nan,
            "along_strike_rmse": math.nan,
            "down_dip_rmse": math.nan,
            "normal_to_plane_rmse": math.nan,
            "along_strike_bias": math.nan,
            "total_missing_lags": math.nan,
            "missing_lags_along_strike": math.nan,
        }
    df = pd.read_csv(variogram_csv)
    df = df[(df["space"] == "nst") & (df["target_pairs"] > 0)].dropna(subset=["gamma_target", "gamma_sim_mean"])
    if df.empty:
        return {
            "weighted_rmse": math.nan,
            "along_strike_rmse": math.nan,
            "down_dip_rmse": math.nan,
            "normal_to_plane_rmse": math.nan,
            "along_strike_bias": math.nan,
            "total_missing_lags": math.nan,
            "missing_lags_along_strike": math.nan,
        }
    weights = {"along_strike": 0.45, "down_dip": 0.40, "normal_to_plane": 0.15}
    out = {
        "weighted_rmse": math.nan,
        "along_strike_rmse": math.nan,
        "down_dip_rmse": math.nan,
        "normal_to_plane_rmse": math.nan,
        "along_strike_bias": math.nan,
        "total_missing_lags": math.nan,
        "missing_lags_along_strike": math.nan,
    }
    wsum = 0.0
    total = 0.0
    for d, w in weights.items():
        sub = df[df["direction"] == d]
        if sub.empty:
            continue
        diff = sub["gamma_sim_mean"].to_numpy(dtype=float) - sub["gamma_target"].to_numpy(dtype=float)
        rmse = float(np.sqrt(np.mean(diff**2)))
        out[f"{d}_rmse"] = rmse
        if d == "along_strike":
            out["along_strike_bias"] = float(np.mean(diff))
        total += w * rmse
        wsum += w
    out["weighted_rmse"] = total / wsum if wsum > 0 else math.nan
    return out


def load_variogram_summary(output_dir: Path) -> dict:
    summary_path = output_dir / "tables" / "variogram_reproduction_summary.json"
    if summary_path.exists():
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        dm = data.get("direction_metrics", {})
        return {
            "weighted_rmse": float(data.get("weighted_rmse", math.nan)),
            "along_strike_rmse": float(data.get("along_strike_rmse", math.nan)),
            "down_dip_rmse": float(dm.get("down_dip", {}).get("rmse", math.nan)),
            "normal_to_plane_rmse": float(dm.get("normal_to_plane", {}).get("rmse", math.nan)),
            "along_strike_bias": float(dm.get("along_strike", {}).get("bias", math.nan)),
            "total_missing_lags": int(data.get("total_missing_lags", 0)),
            "missing_lags_along_strike": int(dm.get("along_strike", {}).get("missing_lags", 0)),
            "lag_coverage_along_strike_pct": float(dm.get("along_strike", {}).get("lag_coverage_pct", math.nan)),
            "lag_coverage_down_dip_pct": float(dm.get("down_dip", {}).get("lag_coverage_pct", math.nan)),
            "lag_coverage_normal_pct": float(dm.get("normal_to_plane", {}).get("lag_coverage_pct", math.nan)),
            "acceptance_all_pass": bool(data.get("acceptance", {}).get("all_pass", False)),
        }
    csv_path = output_dir / "tables" / "variogram_reproduction.csv"
    return variogram_components_from_csv(csv_path)


def score_candidate(metrics: dict, vario: dict, score_profile: str) -> tuple[float, bool, list[str]]:
    weighted = float(vario.get("weighted_rmse", math.nan))
    along = float(vario.get("along_strike_rmse", math.nan))
    if not math.isfinite(weighted):
        return 999.0, False, ["variogram_weighted_rmse_missing"]

    ok, failures = guardrail_status(metrics)
    penalty = 0.0

    # Keep variogram as primary objective; penalize weak along-strike and support gaps.
    if math.isfinite(along) and along > 0.50:
        penalty += 1.5 * (along - 0.50)
    missing = float(vario.get("total_missing_lags", 0.0))
    if missing > 2:
        penalty += 0.05 * (missing - 2.0)
    along_bias = float(vario.get("along_strike_bias", 0.0))
    if math.isfinite(along_bias) and along_bias < -0.20:
        penalty += 0.7 * abs(along_bias + 0.20)

    # Guardrails remain constraints with finite penalties.
    if not ok:
        penalty += 5.0 * len(failures)
    qq = float(metrics.get("qq_rmse", 0.0))
    if qq > GUARDRAILS["qq_rmse_max"]:
        penalty += (qq - GUARDRAILS["qq_rmse_max"]) * 20.0
    hist = float(metrics.get("hist_overlap", 1.0))
    if hist < GUARDRAILS["hist_overlap_min"]:
        penalty += (GUARDRAILS["hist_overlap_min"] - hist) * 20.0

    if score_profile == "balanced":
        swath = np.mean(
            [
                float(metrics.get("swath_corr_x", 0.0)),
                float(metrics.get("swath_corr_y", 0.0)),
                float(metrics.get("swath_corr_z", 0.0)),
            ]
        )
        penalty += 0.4 * max(0.0, 0.55 - swath)

    return weighted + penalty, ok, failures


def build_candidates(base_cfg: dict, max_candidates: int, seed: int = 1337) -> list[dict]:
    all_candidates: list[dict] = []
    for tr, ng, nr, sr, nb in itertools.product(
        TARGET_RANGES,
        NUGGET_RATIOS,
        NORMAL_RANGES,
        SEARCH_RADII,
        NEIGHBOR_BOUNDS,
    ):
        cfg = copy.deepcopy(base_cfg)
        cfg.setdefault("variogram", {}).setdefault("tuning", {})["enabled"] = True
        cfg["variogram"]["tuning"]["target_range_m"] = float(tr)
        cfg["variogram"]["tuning"]["nugget_ratio"] = float(ng)
        cfg.setdefault("variogram", {}).setdefault("anisotropy", {}).setdefault("ranges_m", {})["normal"] = int(nr)
        cfg.setdefault("simulation", {})["search_radius_m"] = [int(sr[0]), int(sr[1]), int(sr[2])]
        cfg["simulation"]["min_neighbors"] = int(nb[0])
        cfg["simulation"]["max_neighbors"] = int(nb[1])
        cfg.setdefault("sensitivity", {})["enabled"] = False
        all_candidates.append(
            {
                "config": cfg,
                "target_range_m": float(tr),
                "nugget_ratio": float(ng),
                "normal_range_m": int(nr),
                "search_radius_m": [int(sr[0]), int(sr[1]), int(sr[2])],
                "min_neighbors": int(nb[0]),
                "max_neighbors": int(nb[1]),
            }
        )
    if len(all_candidates) <= max_candidates:
        return all_candidates
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(all_candidates), size=max_candidates, replace=False))
    return [all_candidates[int(i)] for i in idx]


def candidate_tag(prefix: str, idx: int, c: dict) -> str:
    tr = int(c["target_range_m"])
    ng = int(round(float(c["nugget_ratio"]) * 100))
    nr = int(c["normal_range_m"])
    sr = c["search_radius_m"]
    return (
        f"{prefix}_{idx:03d}_tr{tr}_ng{ng:02d}_nr{nr}"
        f"_sr{int(sr[0])}_{int(sr[1])}_{int(sr[2])}"
        f"_nb{int(c['min_neighbors'])}_{int(c['max_neighbors'])}"
    )


def evaluate_candidate(
    candidate: dict,
    cfg_path: Path,
    out_dir: Path,
    n_real_eval: int,
    min_pairs: int,
    support_aware: bool,
    score_profile: str,
) -> dict:
    row = {
        "tag": cfg_path.stem,
        "config_path": str(cfg_path),
        "output_dir": str(out_dir),
        "target_range_m": candidate["target_range_m"],
        "nugget_ratio": candidate["nugget_ratio"],
        "normal_range_m": candidate["normal_range_m"],
        "search_radius": candidate["search_radius_m"],
        "min_neighbors": candidate["min_neighbors"],
        "max_neighbors": candidate["max_neighbors"],
    }

    ok, err = run_pipeline(cfg_path, out_dir)
    row["ok"] = bool(ok)
    row["run_status"] = "ok" if ok else "failed"
    row["error_reason"] = err
    if not ok:
        row["variogram_repro_ok"] = False
        row["score"] = 999.0
        row["guardrails_ok"] = False
        row["guardrail_failures"] = "run_failed"
        return row

    v_ok, v_err = run_variogram_reproduction(cfg_path, out_dir, n_real_eval, min_pairs, support_aware)
    row["variogram_repro_ok"] = bool(v_ok)
    row["variogram_error_reason"] = v_err
    if not v_ok:
        row["score"] = 999.0
        row["guardrails_ok"] = False
        row["guardrail_failures"] = "variogram_repro_failed"
        return row

    metrics_path = out_dir / "tables" / "validation_metrics.json"
    if not metrics_path.exists():
        row["score"] = 999.0
        row["guardrails_ok"] = False
        row["guardrail_failures"] = "validation_metrics_missing"
        return row
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    vario = load_variogram_summary(out_dir)
    score, guardrails_ok, guardrail_failures = score_candidate(metrics, vario, score_profile=score_profile)

    row.update(metrics)
    row.update(vario)
    row["guardrails_ok"] = bool(guardrails_ok)
    row["guardrail_failures"] = "|".join(guardrail_failures)
    row["score"] = float(score)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize variogram fit with support-aware diagnostics")
    parser.add_argument("--n-real-screen", type=int, default=40)
    parser.add_argument("--n-real-confirm", type=int, default=400)
    parser.add_argument("--max-candidates", type=int, default=36)
    parser.add_argument("--top-confirm", type=int, default=3)
    parser.add_argument("--score-profile", choices=["variogram_first", "balanced"], default="variogram_first")
    parser.add_argument("--min-pairs", type=int, default=200)
    parser.add_argument("--support-aware", dest="support_aware", action="store_true")
    parser.add_argument("--no-support-aware", dest="support_aware", action="store_false")
    parser.set_defaults(support_aware=True)
    args = parser.parse_args()

    base_cfg = load_yaml(CONFIG_PATH)
    candidates = build_candidates(base_cfg, max_candidates=max(1, int(args.max_candidates)))

    TMP_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    screen_rows: list[dict] = []
    print(f"Screening {len(candidates)} candidates...")
    for i, c in enumerate(candidates, start=1):
        cfg = copy.deepcopy(c["config"])
        cfg.setdefault("simulation", {})["n_real"] = int(args.n_real_screen)
        cfg.setdefault("ci", {})["n_real"] = 80
        tag = candidate_tag("screen", i, c)
        cfg_path = TMP_CONFIG_DIR / f"{tag}.yaml"
        out_dir = OUTPUT_ROOT / tag
        save_yaml(cfg_path, cfg)
        row = evaluate_candidate(
            candidate=c,
            cfg_path=cfg_path,
            out_dir=out_dir,
            n_real_eval=min(10, int(args.n_real_screen)),
            min_pairs=int(args.min_pairs),
            support_aware=bool(args.support_aware),
            score_profile=args.score_profile,
        )
        screen_rows.append(row)
        print(f"{tag}: ok={row.get('ok')} score={row.get('score')}")

    screen_df = pd.DataFrame(screen_rows).sort_values("score", ascending=True)
    screen_df.to_csv(OUTPUT_ROOT / "screen_results.csv", index=False)

    eligible = screen_df[
        (screen_df["ok"] == True)
        & (screen_df["variogram_repro_ok"] == True)
        & np.isfinite(screen_df["score"])
    ].copy()
    top = eligible.head(max(1, int(args.top_confirm)))
    if top.empty:
        raise RuntimeError("No successful screening run found.")

    confirm_rows: list[dict] = []
    print(f"Confirming {len(top)} candidates with n_real={int(args.n_real_confirm)}...")
    for rank, (_, srow) in enumerate(top.iterrows(), start=1):
        source_cfg = Path(str(srow["config_path"]))
        cfg = load_yaml(source_cfg)
        cfg.setdefault("simulation", {})["n_real"] = int(args.n_real_confirm)
        cfg.setdefault("ci", {})["n_real"] = 80
        c = {
            "target_range_m": float(srow["target_range_m"]),
            "nugget_ratio": float(srow["nugget_ratio"]),
            "normal_range_m": int(srow["normal_range_m"]),
            "search_radius_m": json.loads(str(srow["search_radius"]).replace("'", '"'))
            if isinstance(srow["search_radius"], str)
            else list(srow["search_radius"]),
            "min_neighbors": int(srow["min_neighbors"]),
            "max_neighbors": int(srow["max_neighbors"]),
        }
        tag = candidate_tag("confirm", rank, c)
        cfg_path = TMP_CONFIG_DIR / f"{tag}.yaml"
        out_dir = OUTPUT_ROOT / tag
        save_yaml(cfg_path, cfg)
        row = evaluate_candidate(
            candidate=c,
            cfg_path=cfg_path,
            out_dir=out_dir,
            n_real_eval=20,
            min_pairs=int(args.min_pairs),
            support_aware=bool(args.support_aware),
            score_profile=args.score_profile,
        )
        confirm_rows.append(row)
        print(f"{tag}: ok={row.get('ok')} score={row.get('score')}")

    confirm_df = pd.DataFrame(confirm_rows).sort_values("score", ascending=True)
    confirm_df.to_csv(OUTPUT_ROOT / "confirm_results.csv", index=False)

    best_pool = confirm_df[
        (confirm_df["ok"] == True)
        & (confirm_df["variogram_repro_ok"] == True)
        & np.isfinite(confirm_df["score"])
    ].copy()
    if best_pool.empty:
        raise RuntimeError("No successful confirm run found.")
    best = best_pool.iloc[0].to_dict()
    best_cfg = load_yaml(Path(str(best["config_path"])))
    best_cfg = enforce_production_invariants(best_cfg)
    save_yaml(BEST_CFG_PATH, best_cfg)
    save_yaml(CONFIG_PATH, best_cfg)

    summary = {
        "best_tag": best["tag"],
        "best_score": float(best["score"]),
        "best_output_dir": str(Path(str(best["output_dir"])).resolve()),
        "best_target_range_m": float(best["target_range_m"]),
        "best_nugget_ratio": float(best["nugget_ratio"]),
        "best_normal_range_m": int(best["normal_range_m"]),
        "best_search_radius_m": best["search_radius"],
        "best_neighbors": {"min": int(best["min_neighbors"]), "max": int(best["max_neighbors"])},
        "best_weighted_rmse": float(best.get("weighted_rmse", math.nan)),
        "best_along_strike_rmse": float(best.get("along_strike_rmse", math.nan)),
        "objective": f"{args.score_profile}_with_guardrails",
        "guardrails": GUARDRAILS,
        "best_config_path": str(BEST_CFG_PATH),
        "applied_to_project_yaml": True,
        "screen_results_csv": str(OUTPUT_ROOT / "screen_results.csv"),
        "confirm_results_csv": str(OUTPUT_ROOT / "confirm_results.csv"),
    }
    BEST_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    manifest = {
        "search_space": {
            "target_range_m": TARGET_RANGES,
            "nugget_ratio": NUGGET_RATIOS,
            "normal_ranges_m": NORMAL_RANGES,
            "search_radius_m": SEARCH_RADII,
            "neighbor_bounds": [{"min": a, "max": b} for a, b in NEIGHBOR_BOUNDS],
        },
        "staging": {
            "screen_n_real": int(args.n_real_screen),
            "confirm_n_real": int(args.n_real_confirm),
            "max_candidates": int(args.max_candidates),
            "top_confirm": int(args.top_confirm),
            "score_profile": args.score_profile,
            "support_aware": bool(args.support_aware),
            "min_pairs": int(args.min_pairs),
        },
        "summary": summary,
    }
    OPT_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Best configuration applied to config/main_config.yaml")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
