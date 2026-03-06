import json
from pathlib import Path

import numpy as np
import pandas as pd


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check_validation(outputs_dir="outputs"):
    outputs = Path(outputs_dir)
    grids_dir = outputs / "grids"
    tables_dir = outputs / "tables"

    reals_path = grids_dir / "sgs_reals.npy"
    if not reals_path.exists():
        raise FileNotFoundError(f"Missing realizations file: {reals_path}")

    domain_path = outputs / "domain_data.csv"
    if not domain_path.exists():
        raise FileNotFoundError(f"Missing domain data file: {domain_path}")

    meta_path = grids_dir / "sgs_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing grid metadata file: {meta_path}")

    print("Loading simulation results...")
    reals = np.load(reals_path)
    meta = _load_json(meta_path)

    print("Loading domain data...")
    data = pd.read_csv(domain_path)

    z_origin = float(meta["z_min"])
    dz = float(meta["dz"])
    nz = int(meta["nz"])

    print(f"Data count: {len(data)}")
    print(f"Model shape: {reals.shape}")

    # Z-slice swath check: domain data vs P50 realization mean profile
    z_bins = np.arange(z_origin, z_origin + (nz + 1) * dz, dz)
    data["z_bin"] = np.digitize(data["z"], z_bins) - 1

    data_z_mean = data.groupby("z_bin")["tgc_pct"].mean()
    model_z_means = np.mean(reals, axis=(1, 2))  # (n_real, nz)
    model_p50_z = np.percentile(model_z_means, 50, axis=0)

    valid_z = data_z_mean.index.to_numpy()
    valid_z = valid_z[(valid_z >= 0) & (valid_z < nz)]

    print("\n--- Validation Report ---")
    if len(valid_z) > 1:
        common_data = data_z_mean.loc[valid_z].to_numpy()
        common_model = model_p50_z[valid_z]
        corr = float(np.corrcoef(common_data, common_model)[0, 1])
        print(f"1. P50 Tracking (Z-Swath Correlation): {corr:.4f}")
        if corr > 0.6:
            print("   -> Good tracking of vertical trends.")
        else:
            print("   -> Weak/moderate tracking. Review normal-direction continuity assumptions.")
    else:
        print("1. P50 Tracking: insufficient overlapping Z bins for correlation.")

    global_max_data = float(data["tgc_pct"].max())
    global_max_model = float(np.max(reals))
    print("\n2. High-Grade Peaks:")
    print(f"   Max Input Grade: {global_max_data:.2f}%")
    print(f"   Max Model Grade: {global_max_model:.2f}%")
    if global_max_model >= global_max_data * 0.9:
        print("   -> Peaks are reasonably captured.")
    else:
        print("   -> Peaks appear smoothed relative to input composites.")

    var_data_z = float(np.var(data_z_mean.to_numpy()))
    var_model_z = float(np.var(model_p50_z))
    ratio = var_model_z / var_data_z if var_data_z > 0 else 0.0
    print("\n3. Vertical Smoothing:")
    print(f"   Variance of Data Z-Means:  {var_data_z:.4f}")
    print(f"   Variance of Model Z-Means: {var_model_z:.4f}")
    print(f"   Smoothing Ratio (Model/Data): {ratio:.2f}")

    metrics_path = tables_dir / "validation_metrics.json"
    if metrics_path.exists():
        metrics = _load_json(metrics_path)
        print("\n4. Saved Validation Metrics Snapshot:")
        print(f"   Histogram overlap: {metrics.get('hist_overlap', float('nan')):.4f}")
        print(f"   QQ RMSE: {metrics.get('qq_rmse', float('nan')):.4f}")
        print(f"   Swath corr X/Y/Z: {metrics.get('swath_corr_x', float('nan')):.4f} / "
              f"{metrics.get('swath_corr_y', float('nan')):.4f} / {metrics.get('swath_corr_z', float('nan')):.4f}")


if __name__ == "__main__":
    check_validation()
