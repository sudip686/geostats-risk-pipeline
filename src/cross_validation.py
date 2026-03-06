"""Ordinary kriging cross-validation metrics (NS-space, with back-transform)."""

import os
import numpy as np
import pandas as pd
import json

from src.variography import run as run_variography
from src.utils.io import load_config
from src.normal_score import NormalScoreTransform
from src.trend import fit_linear_trend, apply_linear_trend
import gstools as gs


def _kfold_indices(n, k=5, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    return np.array_split(idx, k)


def _blocked_kfold_indices(df, k=5, seed=42, block_size_xy=500.0):
    """
    Spatial blocked folds using XY block IDs.
    Blocks (not points) are assigned to folds to reduce spatial leakage.
    """
    if 'x' not in df.columns or 'y' not in df.columns:
        raise KeyError("Blocked CV requires x and y columns")

    x = df['x'].to_numpy(dtype=float)
    y = df['y'].to_numpy(dtype=float)
    min_x = np.min(x)
    min_y = np.min(y)

    bx = np.floor((x - min_x) / float(block_size_xy)).astype(int)
    by = np.floor((y - min_y) / float(block_size_xy)).astype(int)
    block_ids = np.array([f"{ix}_{iy}" for ix, iy in zip(bx, by)])

    uniq_blocks = np.unique(block_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq_blocks)

    # Round-robin assign blocks to folds
    block_to_fold = {bid: (i % k) for i, bid in enumerate(uniq_blocks)}
    fold_id = np.array([block_to_fold[bid] for bid in block_ids], dtype=int)

    folds = [np.where(fold_id == i)[0] for i in range(k)]
    # Guard against empty folds in sparse layouts
    if any(len(f) == 0 for f in folds):
        return _kfold_indices(len(df), k=k, seed=seed)
    return folds


def run(
    config_path='config/project.yaml',
    output_dir='outputs',
    max_samples=300,
    k_folds=5,
    seed=42,
    fold_mode='random',
    block_size_xy=500.0,
    output_name='cross_validation.json',
):
    config = load_config(config_path)
    decl_path = os.path.join(output_dir, 'declustered.csv')
    if not os.path.exists(decl_path):
        raise FileNotFoundError(f"Expected declustered.csv at: {decl_path}")
    data = pd.read_csv(decl_path)

    grade_field = config.get('grade_field', 'tgc_pct')
    if grade_field not in data.columns:
        raise KeyError(f"Missing grade field '{grade_field}' in {decl_path}")

    trend_cfg = config.get('trend', {}) or {}
    trend_cols = trend_cfg.get('columns')
    use_trend = bool(trend_cfg.get('enabled') and trend_cols)

    vario_model, _, _ = run_variography(data_path=None, data_dir=config.get('data_dir', 'data'), config=config)

    if len(data) > max_samples:
        data = data.sample(n=max_samples, random_state=seed).reset_index(drop=True)

    coords_all = (data['x'].to_numpy(), data['y'].to_numpy(), data['z'].to_numpy())
    y_all = data[grade_field].to_numpy(dtype=float)
    w_all = data['decluster_weight'].to_numpy(dtype=float) if 'decluster_weight' in data.columns else None

    if fold_mode == 'blocked':
        folds = _blocked_kfold_indices(data, k=k_folds, seed=seed, block_size_xy=block_size_xy)
    else:
        folds = _kfold_indices(len(data), k=k_folds, seed=seed)
    preds = np.full(len(data), np.nan, dtype=float)

    for test_idx in folds:
        train_mask = np.ones(len(data), dtype=bool)
        train_mask[test_idx] = False
        train_idx = np.where(train_mask)[0]

        xtr = data.loc[train_idx, 'x'].to_numpy()
        ytr = data.loc[train_idx, 'y'].to_numpy()
        ztr = data.loc[train_idx, 'z'].to_numpy()
        gtr = data.loc[train_idx, grade_field].to_numpy(dtype=float)
        wtr = data.loc[train_idx, 'decluster_weight'].to_numpy(dtype=float) if w_all is not None else None

        if use_trend:
            coeffs = fit_linear_trend(data.loc[train_idx], trend_cols, grade_field)
            tr_trend = apply_linear_trend(data.loc[train_idx], trend_cols, coeffs)
            te_trend = apply_linear_trend(data.loc[test_idx], trend_cols, coeffs)
            gtr_res = gtr - tr_trend
        else:
            coeffs = None
            te_trend = 0.0
            gtr_res = gtr

        nst = NormalScoreTransform().fit(gtr_res, weights=wtr)
        gtr_ns = nst.transform(gtr_res)

        krige = gs.Krige(
            vario_model,
            cond_pos=(xtr, ytr, ztr),
            cond_val=gtr_ns,
        )

        for i, idx in enumerate(test_idx):
            px, py, pz = coords_all[0][idx], coords_all[1][idx], coords_all[2][idx]
            pred_ns, _ = krige((px, py, pz))
            pred_res = float(nst.back_transform(np.asarray(pred_ns)).ravel()[0])
            if use_trend:
                preds[idx] = pred_res + float(te_trend[i])
            else:
                preds[idx] = pred_res

    valid = np.isfinite(preds) & np.isfinite(y_all)
    errors = preds[valid] - y_all[valid]
    metrics = {
        'n': int(valid.sum()),
        'k_folds': int(k_folds),
        'fold_mode': str(fold_mode),
        'block_size_xy_m': float(block_size_xy) if fold_mode == 'blocked' else None,
        'ME': float(np.mean(errors)),
        'MAE': float(np.mean(np.abs(errors))),
        'RMSE': float(np.sqrt(np.mean(errors**2))),
        'space': 'original_units_via_ns_kriging',
    }

    os.makedirs(os.path.join(output_dir, 'tables'), exist_ok=True)
    with open(os.path.join(output_dir, 'tables', output_name), 'w') as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == '__main__':
    run()
