from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "figures"


def _load_trend_coeffs() -> tuple[float, float]:
    trend_meta = ROOT / "outputs" / "grids" / "trend_meta.json"
    if trend_meta.exists():
        data = json.loads(trend_meta.read_text(encoding="utf-8"))
        coeffs = data.get("coeffs", [])
        if len(coeffs) >= 2:
            return float(coeffs[0]), float(coeffs[1])

    fallback = ROOT / "trend_diagnostics.json"
    if fallback.exists():
        data = json.loads(fallback.read_text(encoding="utf-8"))
        return float(data["trend_coeff_intercept"]), float(data["trend_coeff_z"])

    raise FileNotFoundError("No trend coefficient source found.")


def generate_trend_diagnostic() -> Path:
    df = pd.read_csv(ROOT / "outputs" / "domain_data.csv")
    if not {"z", "tgc_pct"}.issubset(df.columns):
        raise ValueError("domain_data.csv must contain 'z' and 'tgc_pct'.")

    intercept, slope = _load_trend_coeffs()
    z = df["z"].to_numpy(dtype=float)
    grade = df["tgc_pct"].to_numpy(dtype=float)
    trend = intercept + slope * z
    residual = grade - trend

    # Bin by depth and summarize means for stable visual diagnostics.
    n_bins = 20
    edges = np.linspace(np.nanmin(z), np.nanmax(z), n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    raw_bin = np.full(n_bins, np.nan)
    res_bin = np.full(n_bins, np.nan)

    for i in range(n_bins):
        mask = (z >= edges[i]) & (z < edges[i + 1]) if i < n_bins - 1 else (z >= edges[i]) & (z <= edges[i + 1])
        if np.any(mask):
            raw_bin[i] = np.nanmean(grade[mask])
            res_bin[i] = np.nanmean(residual[mask])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)

    axes[0].scatter(z, grade, s=8, alpha=0.15, color="#1f77b4", label="Composites")
    order = np.argsort(z)
    axes[0].plot(z[order], trend[order], color="#d62728", linewidth=2.0, label="Fitted linear trend")
    axes[0].plot(centers, raw_bin, color="#111111", linewidth=2.2, label="Binned mean")
    axes[0].set_title("Grade vs Z (Before Detrending)")
    axes[0].set_xlabel("Z")
    axes[0].set_ylabel("TGC (%)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best", fontsize=8)

    axes[1].scatter(z, residual, s=8, alpha=0.15, color="#2ca02c", label="Residuals")
    axes[1].axhline(0.0, color="#d62728", linewidth=1.8, label="Zero trend")
    axes[1].plot(centers, res_bin, color="#111111", linewidth=2.2, label="Binned residual mean")
    axes[1].set_title("Residual vs Z (After Detrending)")
    axes[1].set_xlabel("Z")
    axes[1].set_ylabel("Residual TGC (%)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best", fontsize=8)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "trend_diagnostic.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def generate_composite_length_hist() -> Path:
    df = pd.read_csv(ROOT / "outputs" / "composites.csv")
    if "length" in df.columns:
        lengths = df["length"].to_numpy(dtype=float)
    elif {"from_m", "to_m"}.issubset(df.columns):
        lengths = (df["to_m"] - df["from_m"]).to_numpy(dtype=float)
    else:
        raise ValueError("composites.csv must contain 'length' or both 'from_m' and 'to_m'.")

    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    ax.hist(lengths, bins=30, color="#4e79a7", edgecolor="white", alpha=0.9)
    ax.axvline(np.nanmean(lengths), color="#d62728", linewidth=1.8, linestyle="--", label=f"Mean = {np.nanmean(lengths):.2f} m")
    ax.set_title("Composite Length Distribution")
    ax.set_xlabel("Composite length (m)")
    ax.set_ylabel("Frequency")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(loc="best", fontsize=8)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "composite_length_hist.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def main() -> None:
    t = generate_trend_diagnostic()
    c = generate_composite_length_hist()
    print(f"Generated: {t}")
    print(f"Generated: {c}")


if __name__ == "__main__":
    main()
