from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "review_closure"
BASE_CFG = ROOT / "config" / "main_config.yaml"
LOCK = OUT / ".run_lock"
REGISTRY = OUT / "run_registry.json"
SUMMARY_JSON = OUT / "review_summary.json"

REQUIRED_ARTIFACTS = [
    "sgs_meta.json",
    "figures/variogram_model.json",
    "figures/variogram_reproduction.png",
    "tables/risked_tonnage.csv",
    "tables/risked_tonnage_by_realization.csv",
    "tables/validation_metrics.json",
    "tables/cross_validation_blocked_500.json",
    "tables/cross_validation_blocked_500_folds.csv",
]

SCENARIOS: list[tuple[str, dict]] = [
    ("baseline_combined", {}),
    ("domain_fresh", {"target_lith_codes": ["GRSC", "GRSC1", "GRSC2"]}),
    ("domain_weathered", {"target_lith_codes": ["SAP (GRSC)", "SAPR (GRSC)"]}),
    ("calibration_off", {"calibration": {"enabled": False}}),
    ("calibration_on", {"calibration": {"enabled": True}}),
    ("topcut_99_0", {"top_cut": {"enabled": True, "quantile": 99.0}}),
    ("topcut_99_5", {"top_cut": {"enabled": True, "quantile": 99.5}}),
    ("kriging_ok", {"simulation": {"kriging_type": "OK"}}),
    ("kriging_sk", {"simulation": {"kriging_type": "SK"}}),
    ("nb_8_24_sr_250_120_70", {"simulation": {"min_neighbors": 8, "max_neighbors": 24, "search_radius_m": [250, 120, 70]}}),
    ("nb_6_18_sr_220_110_60", {"simulation": {"min_neighbors": 6, "max_neighbors": 18, "search_radius_m": [220, 110, 60]}}),
    ("nb_10_30_sr_280_140_80", {"simulation": {"min_neighbors": 10, "max_neighbors": 30, "search_radius_m": [280, 140, 80]}}),
]


@dataclass
class ScenarioResult:
    name: str
    output_dir: Path
    config_path: Path
    variogram_model: dict
    variogram_ranges: dict
    validation: dict
    blocked_cv: dict
    blocked_cv_folds: pd.DataFrame
    risk: pd.DataFrame
    risk_by_real: pd.DataFrame
    pair_counts: pd.DataFrame


def load_registry() -> dict:
    if REGISTRY.exists():
        return json.loads(REGISTRY.read_text(encoding="utf-8"))
    return {"scenarios": {}, "runs": []}


def save_registry(reg: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    REGISTRY.write_text(json.dumps(reg, indent=2), encoding="utf-8")


def merge(a: dict, b: dict) -> None:
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            merge(a[k], v)
        else:
            a[k] = v


def scenario_config(name: str, edits: dict) -> Path:
    cfg = yaml.safe_load(BASE_CFG.read_text(encoding="utf-8"))
    merge(cfg, edits)
    cfg_dir = OUT / "configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    p = cfg_dir / f"{name}.yaml"
    p.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return p


def scenario_dir(name: str) -> Path:
    return OUT / name


def is_complete(name: str) -> tuple[bool, list[str]]:
    sdir = scenario_dir(name)
    missing = [r for r in REQUIRED_ARTIFACTS if not (sdir / r).exists()]
    return (len(missing) == 0), missing


def run_cmd(cmd: list[str]) -> None:
    print("RUN", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def run_scenario(name: str, edits: dict, force: bool, reg: dict) -> ScenarioResult:
    sdir = scenario_dir(name)
    sdir.mkdir(parents=True, exist_ok=True)
    cfg = scenario_config(name, edits)

    done, missing = is_complete(name)
    if done and not force:
        print(f"SKIP {name}: complete")
    else:
        if missing and not force:
            print(f"RESUME {name}: missing {missing}")
        reg["scenarios"].setdefault(name, {})
        reg["scenarios"][name].update({"status": "running", "config": str(cfg)})
        save_registry(reg)

        if not (sdir / "sgs_meta.json").exists() or force:
            run_cmd(["python", "-m", "src.run_all", "--config", str(cfg), "--output", str(sdir)])

        if not (sdir / "figures" / "variogram_reproduction.png").exists() or force:
            run_cmd([
                "python", "src/variogram_reproduction.py",
                "--config", str(cfg),
                "--outputs", str(sdir),
                "--n-real-eval", "20",
                "--max-grid-samples", "2500",
            ])

        if not (sdir / "tables" / "cross_validation_blocked_500.json").exists() or force:
            run_cmd([
                "python", "-c",
                (
                    "from src.cross_validation import run; "
                    f"run(config_path=r'{cfg}', output_dir=r'{sdir}', fold_mode='blocked', block_size_xy=500.0, output_name='cross_validation_blocked_500.json')"
                ),
            ])

    done_after, missing_after = is_complete(name)
    reg["scenarios"].setdefault(name, {})
    reg["scenarios"][name].update(
        {
            "status": "done" if done_after else "partial",
            "missing": missing_after,
            "output_dir": str(sdir),
        }
    )
    save_registry(reg)

    if not done_after:
        raise RuntimeError(f"Scenario {name} incomplete after run: {missing_after}")

    meta = json.loads((sdir / "sgs_meta.json").read_text(encoding="utf-8"))
    vm = json.loads((sdir / "figures" / "variogram_model.json").read_text(encoding="utf-8"))
    validation = json.loads((sdir / "tables" / "validation_metrics.json").read_text(encoding="utf-8"))
    blocked = json.loads((sdir / "tables" / "cross_validation_blocked_500.json").read_text(encoding="utf-8"))
    folds = pd.read_csv(sdir / "tables" / "cross_validation_blocked_500_folds.csv")
    risk = pd.read_csv(sdir / "tables" / "risked_tonnage.csv")
    risk_by = pd.read_csv(sdir / "tables" / "risked_tonnage_by_realization.csv")
    pair = pd.read_csv(sdir / "figures" / "variogram_pair_counts.csv")

    return ScenarioResult(
        name=name,
        output_dir=sdir,
        config_path=cfg,
        variogram_model=vm,
        variogram_ranges=meta.get("variogram_ranges", {}),
        validation=validation,
        blocked_cv=blocked,
        blocked_cv_folds=folds,
        risk=risk,
        risk_by_real=risk_by,
        pair_counts=pair,
    )


def risk_row(risk_df: pd.DataFrame, cutoff: float) -> pd.Series:
    row = risk_df.loc[np.isclose(risk_df["cutoff"], cutoff)]
    if row.empty:
        raise RuntimeError(f"cutoff {cutoff} missing")
    return row.iloc[0]


def build_domain_rows(base: ScenarioResult, fresh: ScenarioResult, weathered: ScenarioResult) -> list[dict]:
    rows = []
    for r in [base, fresh, weathered]:
        p50_3 = risk_row(r.risk, 3.0)["tonnage_p50"] / 1e6
        p50_4 = risk_row(r.risk, 4.0)["tonnage_p50"] / 1e6
        p50_5 = risk_row(r.risk, 5.0)["tonnage_p50"] / 1e6
        b3 = risk_row(base.risk, 3.0)["tonnage_p50"] / 1e6
        b4 = risk_row(base.risk, 4.0)["tonnage_p50"] / 1e6
        b5 = risk_row(base.risk, 5.0)["tonnage_p50"] / 1e6
        rows.append(
            {
                "scenario": r.name,
                "model_type": r.variogram_model.get("model_type", "NA"),
                "ranges_m": f"{r.variogram_ranges.get('along_strike', float('nan')):.1f}/{r.variogram_ranges.get('down_dip', float('nan')):.1f}/{r.variogram_ranges.get('normal_to_plane', float('nan')):.1f}",
                "nugget": float(r.variogram_model.get("nugget", np.nan)),
                "sill": float(r.variogram_model.get("sill", np.nan)),
                "cv_metrics": f"{r.blocked_cv.get('ME', np.nan):.3f}/{r.blocked_cv.get('MAE', np.nan):.3f}/{r.blocked_cv.get('RMSE', np.nan):.3f}",
                "risk_delta": f"{(p50_3-b3):+.2f}/{(p50_4-b4):+.2f}/{(p50_5-b5):+.2f}",
            }
        )
    return rows


def empirical_prob_rows(base: ScenarioResult) -> list[dict]:
    out = []
    for cutoff in [0.0, 2.0, 3.0, 4.0, 5.0, 6.0]:
        rr = risk_row(base.risk, cutoff)
        by = base.risk_by_real.loc[np.isclose(base.risk_by_real["cutoff"], cutoff)]
        p150 = 100.0 * float((by["tonnage"] >= 150e6).mean())
        p200 = 100.0 * float((by["tonnage"] >= 200e6).mean())
        out.append(
            {
                "cutoff": float(cutoff),
                "p10_mt": float(rr["tonnage_p10"] / 1e6),
                "p50_mt": float(rr["tonnage_p50"] / 1e6),
                "p90_mt": float(rr["tonnage_p90"] / 1e6),
                "risk_width_mt": float((rr["tonnage_p90"] - rr["tonnage_p10"]) / 1e6),
                "p_ge_150_pct": p150,
                "p_ge_200_pct": p200,
            }
        )
    return out


def calibration_rows(off: ScenarioResult, on: ScenarioResult) -> list[dict]:
    rows = []
    for label, r in [("Calibration OFF", off), ("Calibration ON", on)]:
        rows.append(
            {
                "setting": label,
                "hist_overlap": float(r.validation.get("hist_overlap", np.nan)),
                "qq_rmse": float(r.validation.get("qq_rmse", np.nan)),
                "swath_corr": f"{r.validation.get('swath_corr_x', np.nan):.4f}/{r.validation.get('swath_corr_y', np.nan):.4f}/{r.validation.get('swath_corr_z', np.nan):.4f}",
                "cv_metrics": f"{r.blocked_cv.get('ME', np.nan):.3f}/{r.blocked_cv.get('MAE', np.nan):.3f}/{r.blocked_cv.get('RMSE', np.nan):.3f}",
            }
        )
    return rows


def topcut_rows(base: ScenarioResult, capped: list[tuple[str, float, ScenarioResult]]) -> list[dict]:
    rows = []
    for name, quantile, r in capped:
        for cutoff in [3.0, 4.0, 5.0]:
            b = risk_row(base.risk, cutoff)
            s = risk_row(r.risk, cutoff)
            by_b = base.risk_by_real.loc[np.isclose(base.risk_by_real["cutoff"], cutoff)]
            by_s = r.risk_by_real.loc[np.isclose(r.risk_by_real["cutoff"], cutoff)]
            p150_b = 100.0 * float((by_b["tonnage"] >= 150e6).mean())
            p150_s = 100.0 * float((by_s["tonnage"] >= 150e6).mean())
            p200_b = 100.0 * float((by_b["tonnage"] >= 200e6).mean())
            p200_s = 100.0 * float((by_s["tonnage"] >= 200e6).mean())
            w_b = b["tonnage_p90"] - b["tonnage_p10"]
            w_s = s["tonnage_p90"] - s["tonnage_p10"]
            rows.append(
                {
                    "scenario": name,
                    "quantile": float(quantile),
                    "cutoff": float(cutoff),
                    "d_p50_pct": 100.0 * float((s["tonnage_p50"] - b["tonnage_p50"]) / b["tonnage_p50"]) if b["tonnage_p50"] else np.nan,
                    "d_width_pct": 100.0 * float((w_s - w_b) / w_b) if w_b else np.nan,
                    "d_p150_pp": float(p150_s - p150_b),
                    "d_p200_pp": float(p200_s - p200_b),
                }
            )
    return rows


def bootstrap_risk_band(base: ScenarioResult) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(1337)
    for cutoff in sorted(base.risk_by_real["cutoff"].unique()):
        vals = base.risk_by_real.loc[np.isclose(base.risk_by_real["cutoff"], cutoff), "tonnage"].to_numpy()
        if len(vals) == 0:
            continue
        boots = []
        for _ in range(300):
            sample = rng.choice(vals, size=len(vals), replace=True)
            boots.append(np.percentile(sample, 50))
        rows.append(
            {
                "cutoff": float(cutoff),
                "p50_bootstrap_lo_mt": float(np.percentile(boots, 5) / 1e6),
                "p50_bootstrap_mid_mt": float(np.percentile(boots, 50) / 1e6),
                "p50_bootstrap_hi_mt": float(np.percentile(boots, 95) / 1e6),
            }
        )
    return pd.DataFrame(rows)


def domain_swath_diagnostics(base: ScenarioResult) -> pd.DataFrame:
    dom = pd.read_csv(base.output_dir / "domain_data.csv")
    dom["group"] = np.where(dom["lith_code"].astype(str).str.contains("SAP", case=False, na=False), "weathered", "fresh")
    out = []
    for grp, gdf in dom.groupby("group"):
        out.append(
            {
                "domain_group": grp,
                "n": int(len(gdf)),
                "mean_tgc": float(gdf["tgc_pct"].mean()),
                "std_tgc": float(gdf["tgc_pct"].std()),
                "x_span_m": float(gdf["x"].max() - gdf["x"].min()),
                "y_span_m": float(gdf["y"].max() - gdf["y"].min()),
                "z_span_m": float(gdf["z"].max() - gdf["z"].min()),
            }
        )
    return pd.DataFrame(out)


def local_stationarity(base: ScenarioResult) -> pd.DataFrame:
    dom = pd.read_csv(base.output_dir / "domain_data.csv").dropna(subset=["x", "y", "z", "tgc_pct"]).copy()
    dom["z_bin"] = pd.qcut(dom["z"], q=4, duplicates="drop")
    out = []
    for zbin, g in dom.groupby("z_bin"):
        if len(g) < 20:
            continue
        sx = np.polyfit(g["x"].to_numpy(), g["tgc_pct"].to_numpy(), 1)[0]
        sy = np.polyfit(g["y"].to_numpy(), g["tgc_pct"].to_numpy(), 1)[0]
        out.append({"z_bin": str(zbin), "n": int(len(g)), "slope_x": float(sx), "slope_y": float(sy)})
    return pd.DataFrame(out)


def count_complete() -> int:
    return sum(1 for name, _ in SCENARIOS if is_complete(name)[0])


def all_required_results(results: dict[str, ScenarioResult]) -> bool:
    needed = {x[0] for x in SCENARIOS}
    return needed.issubset(set(results.keys()))


def build_summary(results: dict[str, ScenarioResult]) -> None:
    base = results["baseline_combined"]
    top_990 = results["topcut_99_0"]
    top_995 = results["topcut_99_5"]
    summary = {
        "domain_evidence_rows": build_domain_rows(base, results["domain_fresh"], results["domain_weathered"]),
        "table9_rows": empirical_prob_rows(base),
        "calibration_rows": calibration_rows(results["calibration_off"], results["calibration_on"]),
        "topcut_rows": topcut_rows(base, [("topcut_99_0", 99.0, top_990), ("topcut_99_5", 99.5, top_995)]),
        "ok_vs_sk": {
            "ok_rmse": float(results["kriging_ok"].blocked_cv.get("RMSE", np.nan)),
            "sk_rmse": float(results["kriging_sk"].blocked_cv.get("RMSE", np.nan)),
            "ok_me": float(results["kriging_ok"].blocked_cv.get("ME", np.nan)),
            "sk_me": float(results["kriging_sk"].blocked_cv.get("ME", np.nan)),
        },
        "neighborhood_sensitivity": [
            {
                "scenario": n,
                "rmse": float(results[n].blocked_cv.get("RMSE", np.nan)),
                "swath_x": float(results[n].validation.get("swath_corr_x", np.nan)),
                "swath_y": float(results[n].validation.get("swath_corr_y", np.nan)),
                "swath_z": float(results[n].validation.get("swath_corr_z", np.nan)),
            }
            for n in ["nb_8_24_sr_250_120_70", "nb_6_18_sr_220_110_60", "nb_10_30_sr_280_140_80"]
        ],
        "variogram_support": {
            "fit_method": "weighted least squares on experimental gamma(h)",
            "lag_m": 50,
            "max_lag_m": 500,
            "pair_counts_csv": str((base.output_dir / "figures" / "variogram_pair_counts.csv").relative_to(ROOT)),
        },
        "topcut_acceptance": {
            "criterion": "No-cap accepted when absolute delta in primary planning metrics <2%",
            "result": "accepted" if all(abs(x["d_p50_pct"]) < 2.0 for x in topcut_rows(base, [("topcut_99_0", 99.0, top_990), ("topcut_99_5", 99.5, top_995)])) else "not_accepted",
        },
        "risk_probability_formula": "P(Tonnage >= threshold | cutoff c) = count(T_r(c) >= threshold)/R, with R=400 realizations",
        "bootstrap_risk_path": str((OUT / "bootstrap_tonnage_risk_band.csv").relative_to(ROOT)),
        "domain_swath_path": str((OUT / "domain_conditioned_swath.csv").relative_to(ROOT)),
        "local_stationarity_path": str((OUT / "local_stationarity_by_z_panel.csv").relative_to(ROOT)),
    }

    bootstrap_risk_band(base).to_csv(OUT / "bootstrap_tonnage_risk_band.csv", index=False)
    domain_swath_diagnostics(base).to_csv(OUT / "domain_conditioned_swath.csv", index=False)
    local_stationarity(base).to_csv(OUT / "local_stationarity_by_z_panel.csv", index=False)
    pd.DataFrame(summary["neighborhood_sensitivity"]).to_csv(OUT / "neighborhood_sensitivity.csv", index=False)
    pd.DataFrame(summary["topcut_rows"]).to_csv(OUT / "topcut_downstream_sensitivity.csv", index=False)
    pd.DataFrame(summary["table9_rows"]).to_csv(OUT / "table9_empirical_probability.csv", index=False)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def acquire_lock() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if LOCK.exists():
        raise RuntimeError(f"Lock exists: {LOCK}. Another run may still be active.")
    LOCK.write_text("locked", encoding="utf-8")


def release_lock() -> None:
    if LOCK.exists():
        LOCK.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume-safe review closure pipeline")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--force-scenario", default=None)
    parser.add_argument("--max-scenarios", type=int, default=None)
    args = parser.parse_args()

    reg = load_registry()
    status = {name: is_complete(name)[0] for name, _ in SCENARIOS}

    if args.audit_only:
        for name, _ in SCENARIOS:
            done, missing = is_complete(name)
            print(f"{name}: {'COMPLETE' if done else 'PENDING'}")
            if not done:
                print("  missing:", ", ".join(missing))
        return

    acquire_lock()
    try:
        results: dict[str, ScenarioResult] = {}
        completed_before = count_complete()
        ran = 0

        for name, edits in SCENARIOS:
            force = args.force_scenario == name
            done, _ = is_complete(name)
            if done and not force:
                results[name] = run_scenario(name, edits, force=False, reg=reg)
                continue

            if args.max_scenarios is not None and ran >= args.max_scenarios:
                break

            before = count_complete()
            results[name] = run_scenario(name, edits, force=force, reg=reg)
            ran += 1
            after = count_complete()

            if after <= before:
                reg["runs"].append({"status": "no_progress", "scenario": name})
                save_registry(reg)
                done2, missing2 = is_complete(name)
                raise RuntimeError(
                    f"No progress after running {name}. complete={done2}, missing={missing2}. Stop and fix root cause before rerun."
                )

        # Load any complete scenarios not in results map.
        for name, edits in SCENARIOS:
            if name not in results and is_complete(name)[0]:
                results[name] = run_scenario(name, edits, force=False, reg=reg)

        if all_required_results(results):
            build_summary(results)
            reg["runs"].append({"status": "ok", "completed_before": completed_before, "completed_after": count_complete(), "summary": str(SUMMARY_JSON)})
            save_registry(reg)
            print("wrote", SUMMARY_JSON)
        else:
            reg["runs"].append({"status": "partial", "completed_before": completed_before, "completed_after": count_complete()})
            save_registry(reg)
            print("Partial completion only. Re-run to continue.")
    finally:
        release_lock()


if __name__ == "__main__":
    main()
