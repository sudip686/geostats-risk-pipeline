from __future__ import annotations

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ZIP = ROOT / "internal" / "Tanga_MRE_2026-01-06 1 (1).zip"
DEFAULT_MODEL_MEMBER = "OneDrive_2026-01-06/Export Final/04 BM/CSV/MODEL_OK.csv"
DEFAULT_BASELINE_DIR = ROOT / "output" / "review_closure" / "baseline_combined"
DEFAULT_CONFIG = ROOT / "config" / "main_config.yaml"
DEFAULT_LEGACY_COMP = ROOT / "internal" / "internal_validation" / "tables" / "model_ok_vs_sgs_tonnage_grade_comparison.csv"
DEFAULT_LEGACY_SUMMARY = ROOT / "internal" / "internal_validation" / "tables" / "model_ok_vs_sgs_summary.json"
DEFAULT_OUT_DIR = ROOT / "review" / "support_consistency_audit"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model_blocks(zip_path: Path, member_path: str, density_default: float) -> pd.DataFrame:
    with ZipFile(zip_path) as zf:
        with zf.open(member_path) as fh:
            raw = pd.read_csv(fh, low_memory=False)

    cols = {c.upper(): c for c in raw.columns}
    required = ["X", "_X", "Y", "_Y", "Z", "_Z", "TGC_%"]
    missing = [name for name in required if name not in cols]
    if missing:
        raise ValueError(f"MODEL_OK.csv missing required columns: {missing}")

    out = raw[
        [
            cols["X"],
            cols["_X"],
            cols["Y"],
            cols["_Y"],
            cols["Z"],
            cols["_Z"],
            cols["TGC_%"],
        ]
    ].copy()
    out.columns = ["x", "sx", "y", "sy", "z", "sz", "grade"]

    density_col = cols.get("DENSITY")
    if density_col:
        out["density"] = pd.to_numeric(raw[density_col], errors="coerce")
    else:
        out["density"] = np.nan

    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["x", "sx", "y", "sy", "z", "sz", "grade"]).copy()
    out["density"] = out["density"].fillna(density_default)
    out["block_volume_m3"] = out["sx"] * out["sy"] * out["sz"]
    out["block_tonnage_t"] = out["block_volume_m3"] * out["density"]
    return out


def compute_model_curve(model: pd.DataFrame, cutoffs: np.ndarray) -> pd.DataFrame:
    rows: list[dict] = []
    total_tonnage_t = float(model["block_tonnage_t"].sum())
    total_volume_m3 = float(model["block_volume_m3"].sum())

    for cutoff in cutoffs:
        mask = model["grade"] >= cutoff
        tonnage_t = float(model.loc[mask, "block_tonnage_t"].sum())
        volume_m3 = float(model.loc[mask, "block_volume_m3"].sum())
        if tonnage_t > 0.0:
            avg_grade = float(np.average(model.loc[mask, "grade"], weights=model.loc[mask, "block_tonnage_t"]))
            contained_t = tonnage_t * avg_grade / 100.0
        else:
            avg_grade = 0.0
            contained_t = 0.0

        rows.append(
            {
                "cutoff": float(cutoff),
                "model_blocks": int(mask.sum()),
                "model_volume_m3": volume_m3,
                "model_tonnage_t": tonnage_t,
                "model_tonnage_mt": tonnage_t / 1e6,
                "model_avg_grade_pct": avg_grade,
                "model_contained_t": contained_t,
                "model_volume_fraction": volume_m3 / total_volume_m3 if total_volume_m3 else np.nan,
                "model_tonnage_fraction": tonnage_t / total_tonnage_t if total_tonnage_t else np.nan,
            }
        )

    return pd.DataFrame(rows)


def aggregate_model_to_sgs_support(model: pd.DataFrame, grid: dict) -> pd.DataFrame:
    x0, y0, z0 = grid["origin_xyz"]
    dx, dy, dz = float(grid["dx"]), float(grid["dy"]), float(grid["dz"])
    nx, ny, nz = int(grid["nx"]), int(grid["ny"]), int(grid["nz"])

    coarse = model.copy()
    coarse["ix"] = np.rint((coarse["x"] - x0) / dx).astype(int)
    coarse["iy"] = np.rint((coarse["y"] - y0) / dy).astype(int)
    coarse["iz"] = np.rint((coarse["z"] - z0) / dz).astype(int)
    coarse = coarse[
        (coarse["ix"] >= 0)
        & (coarse["ix"] < nx)
        & (coarse["iy"] >= 0)
        & (coarse["iy"] < ny)
        & (coarse["iz"] >= 0)
        & (coarse["iz"] < nz)
    ].copy()

    cell_volume = dx * dy * dz
    coarse["grade_tonnage_product"] = coarse["grade"] * coarse["block_tonnage_t"]

    agg = (
        coarse.groupby(["ix", "iy", "iz"], as_index=False)
        .agg(
            fine_blocks=("grade", "size"),
            filled_volume_m3=("block_volume_m3", "sum"),
            tonnage_t=("block_tonnage_t", "sum"),
            grade_tonnage_product=("grade_tonnage_product", "sum"),
        )
        .copy()
    )
    agg["filled_fraction"] = np.where(
        cell_volume > 0.0,
        agg["filled_volume_m3"] / cell_volume,
        np.nan,
    )
    agg["avg_grade_mass_weighted"] = np.where(
        agg["tonnage_t"] > 0.0,
        agg["grade_tonnage_product"] / agg["tonnage_t"],
        np.nan,
    )
    return agg.drop(columns=["grade_tonnage_product"])


def compare_cells_to_sgs_p50(agg: pd.DataFrame, p50_grid: np.ndarray) -> dict:
    model_vals = []
    sgs_vals = []
    for row in agg.itertuples(index=False):
        sgs_val = float(p50_grid[int(row.ix), int(row.iy), int(row.iz)])
        model_vals.append(float(row.avg_grade_mass_weighted))
        sgs_vals.append(sgs_val)

    a = np.asarray(model_vals, dtype=float)
    b = np.asarray(sgs_vals, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    a = a[valid]
    b = b[valid]
    if a.size < 2:
        return {"n_cells": int(a.size)}

    cutoff_mask = a >= 3.0
    return {
        "n_cells": int(a.size),
        "cell_grade_corr": float(np.corrcoef(a, b)[0, 1]),
        "cell_grade_rmse": float(np.sqrt(np.mean((a - b) ** 2))),
        "cell_grade_bias_model_minus_sgs": float(np.mean(a - b)),
        "occupied_cells_above_3pct_fraction": float(np.mean(cutoff_mask)),
        "model_mean_grade_occupied_cells": float(np.mean(a)),
        "sgs_p50_mean_grade_occupied_cells": float(np.mean(b)),
        "mean_filled_fraction_per_occupied_cell": float(agg["filled_fraction"].mean()),
        "median_filled_fraction_per_occupied_cell": float(agg["filled_fraction"].median()),
        "p90_filled_fraction_per_occupied_cell": float(agg["filled_fraction"].quantile(0.9)),
        "max_filled_fraction_per_occupied_cell": float(agg["filled_fraction"].max()),
    }


def build_comparison_table(
    corrected_model_curve: pd.DataFrame,
    sgs_risk_curve: pd.DataFrame,
    legacy_curve: pd.DataFrame | None,
) -> pd.DataFrame:
    comp = corrected_model_curve.merge(
        sgs_risk_curve[
            [
                "cutoff",
                "tonnage_p10",
                "tonnage_p50",
                "tonnage_p90",
                "grade_p10",
                "grade_p50",
                "grade_p90",
            ]
        ],
        on="cutoff",
        how="left",
    )
    comp = comp.rename(
        columns={
            "tonnage_p10": "sgs_risk_tonnage_p10_t",
            "tonnage_p50": "sgs_risk_tonnage_p50_t",
            "tonnage_p90": "sgs_risk_tonnage_p90_t",
            "grade_p10": "sgs_risk_grade_p10_pct",
            "grade_p50": "sgs_risk_grade_p50_pct",
            "grade_p90": "sgs_risk_grade_p90_pct",
        }
    )
    comp["sgs_risk_tonnage_p10_mt"] = comp["sgs_risk_tonnage_p10_t"] / 1e6
    comp["sgs_risk_tonnage_p50_mt"] = comp["sgs_risk_tonnage_p50_t"] / 1e6
    comp["sgs_risk_tonnage_p90_mt"] = comp["sgs_risk_tonnage_p90_t"] / 1e6
    comp["corrected_ratio_model_to_sgs_p50"] = np.where(
        comp["sgs_risk_tonnage_p50_mt"] > 0.0,
        comp["model_tonnage_mt"] / comp["sgs_risk_tonnage_p50_mt"],
        np.nan,
    )
    comp["corrected_grade_delta_model_minus_sgs"] = (
        comp["model_avg_grade_pct"] - comp["sgs_risk_grade_p50_pct"]
    )

    if legacy_curve is not None:
        legacy = legacy_curve[
            [
                "cutoff",
                "model_tonnage_mt",
                "sgs_tonnage_p50_mt",
                "tonnage_ratio_model_to_sgs_p50",
                "model_avg_grade",
                "sgs_grade_p50",
                "grade_delta_model_minus_sgs",
            ]
        ].rename(
            columns={
                "model_tonnage_mt": "legacy_model_tonnage_mt",
                "sgs_tonnage_p50_mt": "legacy_sgs_tonnage_p50_mt_no_rvf",
                "tonnage_ratio_model_to_sgs_p50": "legacy_ratio_model_to_sgs_no_rvf",
                "model_avg_grade": "legacy_model_avg_grade_pct",
                "sgs_grade_p50": "legacy_sgs_grade_p50_pct",
                "grade_delta_model_minus_sgs": "legacy_grade_delta_model_minus_sgs",
            }
        )
        comp = comp.merge(legacy, on="cutoff", how="left")

    return comp


def make_curve_plot(comp: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), constrained_layout=True)

    axes[0].plot(comp["cutoff"], comp["sgs_risk_tonnage_p50_mt"], label="SGS risk P50", color="#1f77b4", linewidth=2)
    axes[0].fill_between(
        comp["cutoff"],
        comp["sgs_risk_tonnage_p10_mt"],
        comp["sgs_risk_tonnage_p90_mt"],
        color="#1f77b4",
        alpha=0.18,
        label="SGS risk P10-P90",
    )
    axes[0].plot(comp["cutoff"], comp["model_tonnage_mt"], label="MODEL_OK explicit volume", color="#d62728", linewidth=2)
    if "legacy_model_tonnage_mt" in comp.columns:
        axes[0].plot(
            comp["cutoff"],
            comp["legacy_model_tonnage_mt"],
            label="Legacy MODEL_OK comparison",
            color="#ff7f0e",
            linestyle="--",
            linewidth=1.6,
        )
    axes[0].set_title("Tonnage Curve Audit")
    axes[0].set_xlabel("Cutoff (% TGC)")
    axes[0].set_ylabel("Tonnage (Mt)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].plot(comp["cutoff"], comp["sgs_risk_grade_p50_pct"], label="SGS risk P50 grade", color="#1f77b4", linewidth=2)
    axes[1].plot(comp["cutoff"], comp["model_avg_grade_pct"], label="MODEL_OK explicit volume", color="#d62728", linewidth=2)
    if "legacy_model_avg_grade_pct" in comp.columns:
        axes[1].plot(
            comp["cutoff"],
            comp["legacy_model_avg_grade_pct"],
            label="Legacy MODEL_OK grade",
            color="#ff7f0e",
            linestyle="--",
            linewidth=1.6,
        )
    axes[1].set_title("Average Grade Above Cutoff")
    axes[1].set_xlabel("Cutoff (% TGC)")
    axes[1].set_ylabel("Grade (% TGC)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best")

    axes[2].plot(
        comp["cutoff"],
        comp["corrected_ratio_model_to_sgs_p50"],
        label="Corrected ratio",
        color="#2ca02c",
        linewidth=2,
    )
    if "legacy_ratio_model_to_sgs_no_rvf" in comp.columns:
        axes[2].plot(
            comp["cutoff"],
            comp["legacy_ratio_model_to_sgs_no_rvf"],
            label="Legacy ratio",
            color="#9467bd",
            linestyle="--",
            linewidth=1.6,
        )
    axes[2].axhline(1.0, color="#444444", linewidth=1.0, linestyle=":")
    axes[2].set_title("MODEL_OK / SGS P50 Ratio")
    axes[2].set_xlabel("Cutoff (% TGC)")
    axes[2].set_ylabel("Ratio")
    axes[2].grid(alpha=0.25)
    axes[2].legend(loc="best")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def write_markdown_report(path: Path, summary: dict) -> None:
    legacy = summary.get("legacy_3pct", {})
    corrected = summary["corrected_3pct"]
    causes = summary["root_causes"]
    cell = summary["coarse_support_grade_comparison"]

    lines = [
        "# Support-Consistency Audit",
        "",
        "## Executive Summary",
        f"- Legacy 3% comparison reported MODEL_OK {legacy.get('legacy_model_tonnage_mt', float('nan')):.3f} Mt versus SGS {legacy.get('legacy_sgs_tonnage_p50_mt_no_rvf', float('nan')):.3f} Mt.",
        f"- Corrected 3% comparison using explicit MODEL_OK block sizes gives {corrected['model_tonnage_mt']:.3f} Mt versus SGS risk P50 {corrected['sgs_risk_tonnage_p50_mt']:.3f} Mt.",
        f"- Corrected 3% average grade is {corrected['model_avg_grade_pct']:.3f}% TGC versus SGS risk P50 {corrected['sgs_risk_grade_p50_pct']:.3f}% TGC.",
        "",
        "## Root Causes",
        f"- Legacy model block volume was inferred as {causes['legacy_inferred_model_block_volume_m3']:.2f} m3, while explicit MODEL_OK size fields imply a mean block volume of {causes['explicit_mean_model_block_volume_m3']:.2f} m3.",
        f"- Total modeled volume rises from {causes['legacy_inferred_total_model_volume_m3']:.0f} m3 to {causes['explicit_total_model_volume_m3']:.0f} m3 when `_X/_Y/_Z` are used.",
        f"- Legacy SGS comparison ignored the configured rock-volume factor of {causes['rock_volume_factor']:.3f}, inflating the SGS tonnage basis by {causes['sgs_no_rvf_to_rvf_factor']:.2f}x.",
        f"- After correction, the 3% tonnage ratio becomes {corrected['model_to_sgs_ratio']:.4f} instead of {legacy.get('legacy_ratio_model_to_sgs_no_rvf', float('nan')):.6f}.",
        "",
        "## Coarse-Support Comparison",
        f"- Occupied SGS cells: {cell['n_cells']} with mean filled fraction {cell['mean_filled_fraction_per_occupied_cell']:.3f}.",
        f"- Coarse-support grade correlation MODEL_OK vs SGS P50: {cell['cell_grade_corr']:.4f}.",
        f"- Coarse-support grade RMSE: {cell['cell_grade_rmse']:.4f} % TGC, bias {cell['cell_grade_bias_model_minus_sgs']:.4f} % TGC.",
        "",
        "## Interpretation",
        "- The legacy mismatch was dominated by implementation inconsistency, not by a geological disagreement in grade tenor.",
        "- Explicit block dimensions and risk-aligned tonnage comparison reconcile the 3% cutoff almost exactly.",
        "- Remaining work for publication is to formalize support semantics, describe the volume-factor role, and document the validation chain end-to-end.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit support and tonnage consistency between MODEL_OK and SGS outputs.")
    parser.add_argument("--model-zip", type=Path, default=DEFAULT_MODEL_ZIP, help="ZIP archive containing MODEL_OK.csv")
    parser.add_argument("--model-member", default=DEFAULT_MODEL_MEMBER, help="Path to MODEL_OK.csv inside the ZIP")
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR, help="Baseline SGS output directory")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Config file")
    parser.add_argument("--legacy-comp", type=Path, default=DEFAULT_LEGACY_COMP, help="Legacy internal validation comparison CSV")
    parser.add_argument("--legacy-summary", type=Path, default=DEFAULT_LEGACY_SUMMARY, help="Legacy internal validation summary JSON")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    baseline_meta = load_json(args.baseline_dir / "sgs_meta.json")
    baseline_grid = baseline_meta["config"]["grid"]
    sgs_risk_curve = pd.read_csv(args.baseline_dir / "tables" / "risked_tonnage.csv")
    p50_grid = np.load(args.baseline_dir / "grids" / "p50_grid.npy")

    model = load_model_blocks(
        zip_path=args.model_zip,
        member_path=args.model_member,
        density_default=float(cfg.get("density_t_per_m3", 2.43)),
    )
    cutoffs = np.asarray(sgs_risk_curve["cutoff"].to_numpy(dtype=float))
    corrected_model_curve = compute_model_curve(model=model, cutoffs=cutoffs)
    legacy_curve = pd.read_csv(args.legacy_comp) if args.legacy_comp.exists() else None
    legacy_summary = load_json(args.legacy_summary) if args.legacy_summary.exists() else None

    comp = build_comparison_table(
        corrected_model_curve=corrected_model_curve,
        sgs_risk_curve=sgs_risk_curve,
        legacy_curve=legacy_curve,
    )
    agg = aggregate_model_to_sgs_support(model=model, grid=baseline_grid)
    coarse_support = compare_cells_to_sgs_p50(agg=agg, p50_grid=p50_grid)

    corrected_3 = comp.loc[comp["cutoff"] == 3.0].iloc[0]
    legacy_3 = None
    if legacy_curve is not None and (legacy_curve["cutoff"] == 3.0).any():
        legacy_3 = legacy_curve.loc[legacy_curve["cutoff"] == 3.0].iloc[0].to_dict()

    explicit_total_volume = float(model["block_volume_m3"].sum())
    explicit_mean_block_volume = float(model["block_volume_m3"].mean())
    legacy_inferred_total_volume = np.nan
    legacy_inferred_block_volume = np.nan
    if legacy_summary is not None:
        inferred = legacy_summary.get("model_inferred_block_spacing_m", {})
        legacy_inferred_block_volume = float(inferred.get("volume_m3", np.nan))
        legacy_inferred_total_volume = legacy_inferred_block_volume * float(
            legacy_summary.get("blocks_in_overlap_extent", len(model))
        )

    sgs_risk_3 = sgs_risk_curve.loc[sgs_risk_curve["cutoff"] == 3.0].iloc[0]
    summary = {
        "inputs": {
            "model_zip": str(args.model_zip),
            "model_member": args.model_member,
            "baseline_dir": str(args.baseline_dir),
            "config": str(args.config),
        },
        "legacy_3pct": {
            "legacy_model_tonnage_mt": float(legacy_3["model_tonnage_mt"]) if legacy_3 else np.nan,
            "legacy_sgs_tonnage_p50_mt_no_rvf": float(legacy_3["sgs_tonnage_p50_mt"]) if legacy_3 else np.nan,
            "legacy_ratio_model_to_sgs_no_rvf": float(legacy_3["tonnage_ratio_model_to_sgs_p50"]) if legacy_3 else np.nan,
            "legacy_model_avg_grade_pct": float(legacy_3["model_avg_grade"]) if legacy_3 else np.nan,
            "legacy_sgs_grade_p50_pct": float(legacy_3["sgs_grade_p50"]) if legacy_3 else np.nan,
        },
        "corrected_3pct": {
            "model_tonnage_mt": float(corrected_3["model_tonnage_mt"]),
            "sgs_risk_tonnage_p50_mt": float(corrected_3["sgs_risk_tonnage_p50_mt"]),
            "model_to_sgs_ratio": float(corrected_3["corrected_ratio_model_to_sgs_p50"]),
            "model_avg_grade_pct": float(corrected_3["model_avg_grade_pct"]),
            "sgs_risk_grade_p50_pct": float(corrected_3["sgs_risk_grade_p50_pct"]),
            "grade_delta_model_minus_sgs_pct": float(corrected_3["corrected_grade_delta_model_minus_sgs"]),
        },
        "root_causes": {
            "legacy_inferred_model_block_volume_m3": legacy_inferred_block_volume,
            "legacy_inferred_total_model_volume_m3": legacy_inferred_total_volume,
            "explicit_mean_model_block_volume_m3": explicit_mean_block_volume,
            "explicit_total_model_volume_m3": explicit_total_volume,
            "explicit_to_legacy_total_volume_factor": float(explicit_total_volume / legacy_inferred_total_volume)
            if legacy_inferred_total_volume and np.isfinite(legacy_inferred_total_volume)
            else np.nan,
            "rock_volume_factor": float(cfg.get("rock_volume_factor", 1.0)),
            "sgs_no_rvf_to_rvf_factor": float(1.0 / cfg.get("rock_volume_factor", 1.0))
            if cfg.get("rock_volume_factor", 0.0)
            else np.nan,
            "sgs_risk_3pct_tonnage_mt": float(sgs_risk_3["tonnage_p50"] / 1e6),
        },
        "coarse_support_grade_comparison": coarse_support,
        "output_files": {
            "comparison_csv": str(args.out_dir / "support_audit_curve_comparison.csv"),
            "summary_json": str(args.out_dir / "support_audit_summary.json"),
            "curve_plot": str(args.out_dir / "support_audit_curve_comparison.png"),
            "report_md": str(args.out_dir / "support_audit_report.md"),
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    comp.to_csv(args.out_dir / "support_audit_curve_comparison.csv", index=False)
    (args.out_dir / "support_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    make_curve_plot(comp=comp, out_path=args.out_dir / "support_audit_curve_comparison.png")
    write_markdown_report(path=args.out_dir / "support_audit_report.md", summary=summary)

    print("Support audit outputs written:")
    for name, path in summary["output_files"].items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
