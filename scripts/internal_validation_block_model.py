from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.postprocess_risk import calculate_tonnage_curve


DEFAULT_MODEL = ROOT / "outputs" / "internal_validation" / "model_input" / "MODEL_OK.csv"


def load_cfg(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def infer_spacing(values: np.ndarray, fallback: float) -> float:
    values = np.sort(np.unique(values[np.isfinite(values)]))
    if values.size < 2:
        return float(fallback)
    diffs = np.diff(values)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return float(fallback)
    return float(np.median(diffs))


def load_model_blocks(model_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(model_csv, low_memory=False)
    cols = {c.upper(): c for c in df.columns}
    required = ["X", "Y", "Z", "TGC_%"]
    missing = [c for c in required if c not in cols]
    if missing:
        raise ValueError(f"MODEL_OK.csv missing required columns: {missing}")

    out = df[[cols["X"], cols["Y"], cols["Z"], cols["TGC_%"]]].copy()
    out.columns = ["x", "y", "z", "grade"]
    out["x"] = pd.to_numeric(out["x"], errors="coerce")
    out["y"] = pd.to_numeric(out["y"], errors="coerce")
    out["z"] = pd.to_numeric(out["z"], errors="coerce")
    out["grade"] = pd.to_numeric(out["grade"], errors="coerce")

    density_col = cols.get("DENSITY")
    if density_col:
        out["density"] = pd.to_numeric(df[density_col], errors="coerce")
    else:
        out["density"] = np.nan

    out = out.dropna(subset=["x", "y", "z", "grade"]).copy()
    return out


def grid_axes_from_meta(meta: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(meta["x_min"], meta["x_max"], int(meta["nx"]))
    y = np.linspace(meta["y_min"], meta["y_max"], int(meta["ny"]))
    z = np.linspace(meta["z_min"], meta["z_max"], int(meta["nz"]))
    return x, y, z


def swath_from_sgs(reals: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # realizations expected shape: (n_real, nx, ny, nz)
    axes_to_mean = tuple(i for i in (1, 2, 3) if i != axis + 1)
    means = np.mean(reals, axis=axes_to_mean)
    p10 = np.percentile(means, 10, axis=0)
    p50 = np.percentile(means, 50, axis=0)
    p90 = np.percentile(means, 90, axis=0)
    return p10, p50, p90


def swath_from_blocks(blocks: pd.DataFrame, axis_name: str, axis_coords: np.ndarray) -> np.ndarray:
    if axis_coords.size < 2:
        return np.full(axis_coords.size, np.nan)
    step = axis_coords[1] - axis_coords[0]
    bins = np.arange(axis_coords[0], axis_coords[-1] + step, step)
    b = blocks.copy()
    b = b[(b[axis_name] >= axis_coords[0]) & (b[axis_name] <= axis_coords[-1])]
    if b.empty:
        return np.full(axis_coords.size, np.nan)
    b["bin"] = np.digitize(b[axis_name], bins) - 1
    means = b.groupby("bin")["grade"].mean()
    idx = np.arange(axis_coords.size)
    return means.reindex(idx).to_numpy()


def make_swath_plot(
    axis_name: str,
    coords: np.ndarray,
    sgs_p10: np.ndarray,
    sgs_p50: np.ndarray,
    sgs_p90: np.ndarray,
    model_curve: np.ndarray,
    out_path: Path,
) -> dict:
    valid = np.isfinite(model_curve)
    corr = float(np.corrcoef(model_curve[valid], sgs_p50[valid])[0, 1]) if np.sum(valid) >= 2 else float("nan")
    rmse = float(np.sqrt(np.mean((model_curve[valid] - sgs_p50[valid]) ** 2))) if np.sum(valid) >= 1 else float("nan")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(coords, sgs_p50, color="#1f77b4", linewidth=2, label="SGS P50 swath")
    ax.fill_between(coords, sgs_p10, sgs_p90, color="#1f77b4", alpha=0.2, label="SGS P10-P90")
    ax.plot(coords[valid], model_curve[valid], "o--", color="#d62728", markersize=3, label="MODEL_OK swath")
    ax.set_title(f"Internal Validation Swath - {axis_name.upper()} direction")
    ax.set_xlabel(f"{axis_name.upper()} coordinate (m)")
    ax.set_ylabel("Grade (% TGC)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    return {"corr": corr, "rmse": rmse, "n_bins": int(np.sum(valid))}


def model_tonnage_curve(blocks: pd.DataFrame, density_default: float, cutoffs: np.ndarray) -> pd.DataFrame:
    dx = infer_spacing(blocks["x"].to_numpy(), 100.0)
    dy = infer_spacing(blocks["y"].to_numpy(), 100.0)
    dz = infer_spacing(blocks["z"].to_numpy(), 10.0)
    vol = dx * dy * dz

    density = blocks["density"].copy()
    density = density.fillna(density_default)
    density = density.replace([np.inf, -np.inf], np.nan).fillna(density_default)

    rows = []
    for c in cutoffs:
        mask = blocks["grade"] >= c
        if not mask.any():
            rows.append(
                {
                    "cutoff": c,
                    "model_blocks": 0,
                    "model_tonnage_t": 0.0,
                    "model_tonnage_mt": 0.0,
                    "model_avg_grade": 0.0,
                    "model_contained_t": 0.0,
                }
            )
            continue
        t = float(np.sum(vol * density[mask]))
        g = float(np.mean(blocks.loc[mask, "grade"]))
        rows.append(
            {
                "cutoff": c,
                "model_blocks": int(mask.sum()),
                "model_tonnage_t": t,
                "model_tonnage_mt": t / 1e6,
                "model_avg_grade": g,
                "model_contained_t": t * g / 100.0,
            }
        )
    out = pd.DataFrame(rows)
    out.attrs["dx"] = dx
    out.attrs["dy"] = dy
    out.attrs["dz"] = dz
    out.attrs["volume"] = vol
    return out


def make_tonnage_grade_plot(comp: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), constrained_layout=True)

    axes[0].plot(comp["cutoff"], comp["model_tonnage_mt"], "r-o", label="MODEL_OK", markersize=3)
    axes[0].plot(comp["cutoff"], comp["sgs_tonnage_p50_mt"], "b-", label="SGS P50")
    axes[0].fill_between(comp["cutoff"], comp["sgs_tonnage_p10_mt"], comp["sgs_tonnage_p90_mt"], color="b", alpha=0.2, label="SGS P10-P90")
    axes[0].set_title("Tonnage Curve Comparison")
    axes[0].set_xlabel("Cutoff (% TGC)")
    axes[0].set_ylabel("Tonnage (Mt)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].plot(comp["cutoff"], comp["model_avg_grade"], "r-o", label="MODEL_OK", markersize=3)
    axes[1].plot(comp["cutoff"], comp["sgs_grade_p50"], "b-", label="SGS P50")
    axes[1].set_title("Average Grade Above Cutoff")
    axes[1].set_xlabel("Cutoff (% TGC)")
    axes[1].set_ylabel("Grade (% TGC)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best")

    axes[2].plot(comp["cutoff"], comp["model_block_fraction"], "r-o", label="MODEL_OK", markersize=3)
    axes[2].plot(comp["cutoff"], comp["sgs_block_fraction_p50"], "b-", label="SGS P50")
    axes[2].fill_between(comp["cutoff"], comp["sgs_block_fraction_p10"], comp["sgs_block_fraction_p90"], color="b", alpha=0.2, label="SGS P10-P90")
    axes[2].set_title("Normalized Block Fraction Above Cutoff")
    axes[2].set_xlabel("Cutoff (% TGC)")
    axes[2].set_ylabel("Fraction of blocks")
    axes[2].grid(alpha=0.25)
    axes[2].legend(loc="best")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def run(model_csv: Path, outputs_dir: Path, config_path: Path) -> None:
    cfg = load_cfg(config_path)
    density_default = float(cfg.get("density_t_per_m3", 2.43))
    out_dir = outputs_dir / "internal_validation"
    fig_dir = out_dir / "figures"
    table_dir = out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    model = load_model_blocks(model_csv)

    meta = json.loads((outputs_dir / "grids" / "sgs_meta.json").read_text(encoding="utf-8"))
    x_axis, y_axis, z_axis = grid_axes_from_meta(meta)

    reals_path = outputs_dir / "grids" / "sgs_reals_calibrated.npy"
    if not reals_path.exists():
        reals_path = outputs_dir / "grids" / "sgs_reals.npy"
    reals = np.load(reals_path)
    if reals.ndim != 4:
        raise ValueError(f"Unexpected SGS realization shape: {reals.shape}")

    # Restrict model to overlapping SGS extent.
    model_clip = model[
        (model["x"] >= x_axis.min()) & (model["x"] <= x_axis.max()) &
        (model["y"] >= y_axis.min()) & (model["y"] <= y_axis.max()) &
        (model["z"] >= z_axis.min()) & (model["z"] <= z_axis.max())
    ].copy()
    if model_clip.empty:
        raise ValueError("No MODEL_OK blocks fall within SGS grid extents.")

    swath_metrics = {}
    for axis_idx, axis_name, coords in [(0, "x", x_axis), (1, "y", y_axis), (2, "z", z_axis)]:
        p10, p50, p90 = swath_from_sgs(reals, axis=axis_idx)
        model_curve = swath_from_blocks(model_clip, axis_name=axis_name, axis_coords=coords)
        m = make_swath_plot(
            axis_name=axis_name,
            coords=coords,
            sgs_p10=p10,
            sgs_p50=p50,
            sgs_p90=p90,
            model_curve=model_curve,
            out_path=fig_dir / f"swath_{axis_name}_model_vs_sgs.png",
        )
        swath_metrics[axis_name] = m

    cutoffs = np.arange(0.0, 21.0, 1.0)
    sgs_curve = calculate_tonnage_curve(
        reals,
        cutoffs=cutoffs,
        volume_per_block=float(meta["dx"] * meta["dy"] * meta["dz"]),
        density=density_default,
        volume_factor=1.0,
    )
    model_curve = model_tonnage_curve(model_clip, density_default=density_default, cutoffs=cutoffs)

    # Normalized block-fraction curves for support-independent comparison.
    sgs_block_fracs = []
    for c in cutoffs:
        per_real = np.mean(reals >= c, axis=(1, 2, 3))
        sgs_block_fracs.append(
            {
                "cutoff": c,
                "sgs_block_fraction_p10": float(np.percentile(per_real, 10)),
                "sgs_block_fraction_p50": float(np.percentile(per_real, 50)),
                "sgs_block_fraction_p90": float(np.percentile(per_real, 90)),
            }
        )
    sgs_frac_df = pd.DataFrame(sgs_block_fracs)
    model_curve["model_block_fraction"] = model_curve["model_blocks"] / float(len(model_clip))

    comp = sgs_curve[["cutoff", "tonnage_p10", "tonnage_p50", "tonnage_p90", "grade_p50"]].copy()
    comp = comp.rename(
        columns={
            "tonnage_p10": "sgs_tonnage_p10_t",
            "tonnage_p50": "sgs_tonnage_p50_t",
            "tonnage_p90": "sgs_tonnage_p90_t",
            "grade_p50": "sgs_grade_p50",
        }
    )
    comp["sgs_tonnage_p10_mt"] = comp["sgs_tonnage_p10_t"] / 1e6
    comp["sgs_tonnage_p50_mt"] = comp["sgs_tonnage_p50_t"] / 1e6
    comp["sgs_tonnage_p90_mt"] = comp["sgs_tonnage_p90_t"] / 1e6
    comp = comp.merge(model_curve, on="cutoff", how="left")
    comp = comp.merge(sgs_frac_df, on="cutoff", how="left")
    comp["tonnage_ratio_model_to_sgs_p50"] = np.where(
        comp["sgs_tonnage_p50_mt"] > 0, comp["model_tonnage_mt"] / comp["sgs_tonnage_p50_mt"], np.nan
    )
    comp["grade_delta_model_minus_sgs"] = comp["model_avg_grade"] - comp["sgs_grade_p50"]
    comp_path = table_dir / "model_ok_vs_sgs_tonnage_grade_comparison.csv"
    try:
        comp.to_csv(comp_path, index=False)
    except PermissionError:
        suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        comp_path = table_dir / f"model_ok_vs_sgs_tonnage_grade_comparison_{suffix}.csv"
        comp.to_csv(comp_path, index=False)

    make_tonnage_grade_plot(comp, fig_dir / "tonnage_grade_curve_model_vs_sgs.png")

    c3 = comp.loc[comp["cutoff"] == 3.0].iloc[0]
    summary = {
        "model_csv": str(model_csv),
        "sgs_realizations_file": reals_path.name,
        "blocks_total_in_model": int(len(model)),
        "blocks_in_overlap_extent": int(len(model_clip)),
        "model_inferred_block_spacing_m": {
            "dx": float(model_curve.attrs["dx"]),
            "dy": float(model_curve.attrs["dy"]),
            "dz": float(model_curve.attrs["dz"]),
            "volume_m3": float(model_curve.attrs["volume"]),
        },
        "swath_metrics": swath_metrics,
        "cutoff_3pct": {
            "model_tonnage_mt": float(c3["model_tonnage_mt"]),
            "sgs_p50_tonnage_mt": float(c3["sgs_tonnage_p50_mt"]),
            "model_to_sgs_tonnage_ratio": float(c3["tonnage_ratio_model_to_sgs_p50"]),
            "model_avg_grade_pct": float(c3["model_avg_grade"]),
            "sgs_p50_grade_pct": float(c3["sgs_grade_p50"]),
            "grade_delta_model_minus_sgs_pct": float(c3["grade_delta_model_minus_sgs"]),
            "model_block_fraction": float(c3["model_block_fraction"]),
            "sgs_block_fraction_p50": float(c3["sgs_block_fraction_p50"]),
        },
        "outputs": {
            "swath_x": str(fig_dir / "swath_x_model_vs_sgs.png"),
            "swath_y": str(fig_dir / "swath_y_model_vs_sgs.png"),
            "swath_z": str(fig_dir / "swath_z_model_vs_sgs.png"),
            "tonnage_grade_curve": str(fig_dir / "tonnage_grade_curve_model_vs_sgs.png"),
            "comparison_table": str(comp_path),
        },
    }
    summary_path = table_dir / "model_ok_vs_sgs_summary.json"
    try:
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    except PermissionError:
        suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = table_dir / f"model_ok_vs_sgs_summary_{suffix}.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Internal validation outputs written:")
    for k, v in summary["outputs"].items():
        print(f"- {k}: {v}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Internal validation: MODEL_OK vs SGS")
    parser.add_argument("--model-csv", type=Path, default=DEFAULT_MODEL, help="Path to MODEL_OK.csv")
    parser.add_argument("--outputs-dir", type=Path, default=ROOT / "outputs", help="Outputs directory root")
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "project_best_fit.yaml", help="Config file")
    args = parser.parse_args()

    run(model_csv=args.model_csv, outputs_dir=args.outputs_dir, config_path=args.config)


if __name__ == "__main__":
    main()
