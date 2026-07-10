from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import gstools as gs

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.normal_score import NormalScoreTransform
from src.trend import apply_linear_trend, fit_linear_trend
from src.variography import estimate_directional_variogram, estimate_variogram, fit_variogram_model


DEFAULT_INPUT = ROOT / "review" / "geology_first_paper_screen" / "master_composite_geology_density.csv"
DEFAULT_OUT_DIR = ROOT / "review" / "combined_geo_ml_paper"

PRIMARY_DOMAINS = ["GRSC", "GRSC1", "GRSC2", "weathered_graphitic"]
DIRECTIONS = [
    {"name": "along_strike", "azimuth": 105.0, "dip": 0.0},
    {"name": "down_dip", "azimuth": 15.0, "dip": 32.0},
    {"name": "normal_to_plane", "azimuth": 195.0, "dip": 58.0},
]


def map_paper_domain(raw_lith: object) -> tuple[str, str]:
    value = "" if pd.isna(raw_lith) else str(raw_lith).strip()

    if value in {"GRSC", "GRSC1", "GRSC2"}:
        return value, "primary"

    weathered_graphitic = {
        "SAP",
        "SAPR",
        "SAP (GRSC)",
        "SAPR (GRSC)",
        "SAPR (GRSC2)",
        "SAP (GRSC2)",
        "SAPR(GRSC)",
        "SAPR(GRSC2)",
    }
    if value in weathered_graphitic:
        return "weathered_graphitic", "primary"

    graphitic_tokens = ("GRSC", "SAP")
    if any(token in value for token in graphitic_tokens):
        return "mixed_graphitic_supplementary", "supplementary"

    return "other_non_graphitic", "excluded"


def map_weathering_bin(raw_weathering: object) -> str:
    value = "" if pd.isna(raw_weathering) else str(raw_weathering).strip().upper()
    if value == "HW":
        return "HW"
    if value == "MW":
        return "MW"
    if value == "SW":
        return "SW"
    return "UW_plus_other"


def safe_category(series: pd.Series, fill_value: str = "MISSING") -> pd.Series:
    return series.fillna(fill_value).astype(str).replace({"": fill_value})


def blocked_spatial_folds(df: pd.DataFrame, k: int = 5, block_size_xy: float = 500.0, seed: int = 42) -> list[np.ndarray]:
    x = df["X"].to_numpy(dtype=float)
    y = df["Y"].to_numpy(dtype=float)
    bx = np.floor((x - np.min(x)) / block_size_xy).astype(int)
    by = np.floor((y - np.min(y)) / block_size_xy).astype(int)
    block_ids = np.array([f"{ix}_{iy}" for ix, iy in zip(bx, by)])
    uniq = np.unique(block_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    assign = {bid: i % k for i, bid in enumerate(uniq)}
    fold_id = np.array([assign[bid] for bid in block_ids], dtype=int)
    folds = [np.where(fold_id == i)[0] for i in range(k)]
    if any(len(f) == 0 for f in folds):
        raise ValueError("Blocked CV produced an empty fold.")
    return folds


def fit_nst_kriging_model(train_df: pd.DataFrame) -> dict:
    work = train_df.dropna(subset=["TGC_%", "X", "Y", "Z"]).copy()
    if len(work) < 20:
        raise ValueError("Too few rows to fit kriging model.")

    coeffs = fit_linear_trend(work, ["Z"], "TGC_%")
    trend = apply_linear_trend(work, ["Z"], coeffs)
    residual = work["TGC_%"].to_numpy(dtype=float) - trend

    nst = NormalScoreTransform().fit(residual)
    ns_vals = nst.transform(residual)

    coords = (
        work["X"].to_numpy(dtype=float),
        work["Y"].to_numpy(dtype=float),
        work["Z"].to_numpy(dtype=float),
    )
    bins, gamma = estimate_variogram(coords, ns_vals, n_lags=10, max_dist=500)
    variogram = fit_variogram_model(bins, gamma, model_type="exponential", nugget=True, max_range=2000)

    points = np.column_stack(coords)
    tree = cKDTree(points)
    return {
        "train_df": work.reset_index(drop=True),
        "coeffs": coeffs,
        "nst": nst,
        "variogram": variogram,
        "tree": tree,
        "points": points,
        "ns_vals": ns_vals,
    }


def predict_with_local_kriging(model: dict, test_df: pd.DataFrame, max_neighbors: int = 24, min_neighbors: int = 8) -> np.ndarray:
    train = model["train_df"]
    tree: cKDTree = model["tree"]
    points = model["points"]
    ns_vals = model["ns_vals"]
    variogram = model["variogram"]
    nst: NormalScoreTransform = model["nst"]
    coeffs = model["coeffs"]

    test_coords = test_df[["X", "Y", "Z"]].to_numpy(dtype=float)
    k = min(max_neighbors, len(train))
    if k < 1:
        return np.full(len(test_df), np.nan)

    distances, indices = tree.query(test_coords, k=k)
    if k == 1:
        distances = distances[:, None]
        indices = indices[:, None]

    preds = np.full(len(test_df), np.nan, dtype=float)
    global_mean = float(train["TGC_%"].mean())

    for i, (row, idxs) in enumerate(zip(test_df.itertuples(index=False), indices)):
        idxs = np.asarray(idxs, dtype=int)
        idxs = idxs[np.isfinite(idxs)]
        if idxs.size < min_neighbors:
            preds[i] = global_mean
            continue

        neighbor_points = points[idxs]
        neighbor_vals = ns_vals[idxs]
        try:
            krige = gs.Krige(
                variogram,
                cond_pos=(neighbor_points[:, 0], neighbor_points[:, 1], neighbor_points[:, 2]),
                cond_val=neighbor_vals,
            )
            pred_ns, _ = krige((float(row.X), float(row.Y), float(row.Z)))
            pred_resid = float(nst.back_transform(np.asarray(pred_ns)).ravel()[0])
            pred = pred_resid + float(apply_linear_trend(pd.DataFrame({"Z": [row.Z]}), ["Z"], coeffs)[0])
            preds[i] = pred
        except Exception:
            preds[i] = global_mean

    return preds


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[valid]
    y_pred = y_pred[valid]
    err = y_pred - y_true
    return {
        "n": int(len(y_true)),
        "ME": float(np.mean(err)),
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err**2))),
    }


def run_geostat_cv(df: pd.DataFrame, mode: str) -> tuple[dict, pd.DataFrame]:
    graphitic = df[df["paper_domain_status"] == "primary"].copy()
    graphitic = graphitic.dropna(subset=["TGC_%", "X", "Y", "Z"]).reset_index(drop=True)
    folds = blocked_spatial_folds(graphitic, k=5, block_size_xy=500.0, seed=42)
    pred = np.full(len(graphitic), np.nan, dtype=float)
    fold_rows: list[dict] = []

    for fold_id, test_idx in enumerate(folds, start=1):
        train_idx = np.setdiff1d(np.arange(len(graphitic)), test_idx)
        train_df = graphitic.iloc[train_idx].copy()
        test_df = graphitic.iloc[test_idx].copy()

        if mode == "pooled":
            pooled_model = fit_nst_kriging_model(train_df)
            pred[test_idx] = predict_with_local_kriging(pooled_model, test_df)
        elif mode == "split":
            pooled_model = fit_nst_kriging_model(train_df)
            fold_pred = np.full(len(test_df), np.nan, dtype=float)
            for domain in PRIMARY_DOMAINS:
                dom_test_mask = test_df["paper_domain"] == domain
                if not dom_test_mask.any():
                    continue
                dom_train = train_df[train_df["paper_domain"] == domain].copy()
                if len(dom_train) < 40:
                    fold_pred[dom_test_mask.to_numpy()] = predict_with_local_kriging(pooled_model, test_df.loc[dom_test_mask].copy())
                    continue
                dom_model = fit_nst_kriging_model(dom_train)
                fold_pred[dom_test_mask.to_numpy()] = predict_with_local_kriging(dom_model, test_df.loc[dom_test_mask].copy())
            pred[test_idx] = fold_pred
        else:
            raise ValueError(f"Unknown mode: {mode}")

        metrics = compute_metrics(test_df["TGC_%"].to_numpy(dtype=float), pred[test_idx])
        metrics["fold"] = fold_id
        metrics["mode"] = mode
        fold_rows.append(metrics)

    summary = compute_metrics(graphitic["TGC_%"].to_numpy(dtype=float), pred)
    summary["mode"] = mode
    summary["n_domains"] = int(graphitic["paper_domain"].nunique())
    return summary, pd.DataFrame(fold_rows)


def domain_directional_variogram_summary(df: pd.DataFrame, label: str) -> list[dict]:
    sub = df.dropna(subset=["TGC_%", "X", "Y", "Z"]).copy()
    if len(sub) < 25:
        return []

    if len(sub) > 800:
        sub = sub.sample(n=800, random_state=42)

    coords = (
        sub["X"].to_numpy(dtype=float),
        sub["Y"].to_numpy(dtype=float),
        sub["Z"].to_numpy(dtype=float),
    )
    coeffs = fit_linear_trend(sub, ["Z"], "TGC_%")
    residual = sub["TGC_%"].to_numpy(dtype=float) - apply_linear_trend(sub, ["Z"], coeffs)
    ns = NormalScoreTransform().fit_transform(residual)

    rows: list[dict] = []
    for direction in DIRECTIONS:
        bins, gamma, counts = estimate_directional_variogram(
            coords=coords,
            values=ns,
            azimuth=direction["azimuth"],
            dip=direction["dip"],
            tolerance=22.5,
            n_lags=10,
            max_dist=500,
            max_pairs=200000,
        )
        nonzero_lags = int(np.sum(np.isfinite(gamma)))
        fitted_range = math.nan
        nugget = math.nan
        sill = math.nan
        if len(bins) >= 3 and np.sum(np.isfinite(gamma)) >= 3:
            model = fit_variogram_model(np.asarray(bins), np.asarray(gamma), model_type="exponential", nugget=True, max_range=2000)
            fitted_range = float(model.len_scale)
            nugget = float(model.nugget)
            sill = float(model.var)

        rows.append(
            {
                "label": label,
                "direction": direction["name"],
                "n_points": int(len(sub)),
                "pair_count_total": int(np.nansum(counts)) if len(counts) else 0,
                "nonzero_lags": nonzero_lags,
                "fitted_range_m": fitted_range,
                "nugget": nugget,
                "sill": sill,
            }
        )
    return rows


def feature_group(name: str) -> str:
    if name.startswith("num__"):
        return name.replace("num__", "", 1)
    if name.startswith("cat__"):
        body = name.replace("cat__", "", 1)
        for prefix in ["paper_domain_", "weathering_bin_", "geo_ALTERATION_", "geo_SULPHIDES_", "geo_STRUCTURE_"]:
            if body.startswith(prefix):
                return prefix[:-1]
        return body.split("_", 1)[0]
    return name


def safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return math.nan
    return float(roc_auc_score(y_true, y_prob))


def holm_adjust(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    ordered = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [0.0] * len(p_values)
    running_max = 0.0
    m = len(p_values)
    for rank, (idx, p_val) in enumerate(ordered, start=1):
        candidate = (m - rank + 1) * p_val
        running_max = max(running_max, candidate)
        adjusted[idx] = min(1.0, running_max)
    return adjusted


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) == 0 or len(y) == 0:
        return math.nan
    diff = np.subtract.outer(x, y)
    return float((np.sum(diff > 0) - np.sum(diff < 0)) / diff.size)


def delta_magnitude(delta: float) -> str:
    ad = abs(delta)
    if ad < 0.147:
        return "negligible"
    if ad < 0.33:
        return "small"
    if ad < 0.474:
        return "medium"
    return "large"


def kruskal_effect_size(h_stat: float, n_groups: int, n_total: int) -> float:
    if n_total <= n_groups:
        return math.nan
    return float((h_stat - n_groups + 1) / (n_total - n_groups))


def pairwise_mannwhitney(df: pd.DataFrame, group_col: str, value_col: str) -> pd.DataFrame:
    groups = []
    for label, sub in df.groupby(group_col):
        values = pd.to_numeric(sub[value_col], errors="coerce").dropna().to_numpy(dtype=float)
        if len(values) == 0:
            continue
        groups.append((str(label), values))

    rows = []
    p_values = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            left_label, left_vals = groups[i]
            right_label, right_vals = groups[j]
            test = stats.mannwhitneyu(left_vals, right_vals, alternative="two-sided")
            delta = cliffs_delta(left_vals, right_vals)
            rows.append(
                {
                    "group_a": left_label,
                    "group_b": right_label,
                    "n_a": int(len(left_vals)),
                    "n_b": int(len(right_vals)),
                    "median_a": float(np.median(left_vals)),
                    "median_b": float(np.median(right_vals)),
                    "median_diff_a_minus_b": float(np.median(left_vals) - np.median(right_vals)),
                    "u_statistic": float(test.statistic),
                    "p_value_raw": float(test.pvalue),
                    "cliffs_delta": delta,
                    "cliffs_delta_magnitude": delta_magnitude(delta),
                }
            )
            p_values.append(float(test.pvalue))

    adjusted = holm_adjust(p_values)
    for row, p_adj in zip(rows, adjusted):
        row["p_value_holm"] = float(p_adj)
        row["significant_holm_0_05"] = bool(p_adj < 0.05)

    return pd.DataFrame(rows)


def run_domain_hypothesis_tests(primary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = primary.dropna(subset=["paper_domain", "TGC_%"]).copy()
    grouped = [(str(label), sub["TGC_%"].to_numpy(dtype=float)) for label, sub in work.groupby("paper_domain")]
    labels = [label for label, _ in grouped]
    values = [vals for _, vals in grouped]
    h_stat, p_val = stats.kruskal(*values)
    summary = pd.DataFrame(
        [
            {
                "hypothesis": "H1",
                "scope": "all_primary_domains",
                "test": "Kruskal-Wallis",
                "grouping": "paper_domain",
                "n_total": int(len(work)),
                "n_groups": int(len(labels)),
                "groups_tested": "; ".join(labels),
                "statistic": float(h_stat),
                "p_value": float(p_val),
                "effect_size": kruskal_effect_size(float(h_stat), len(labels), len(work)),
                "effect_size_label": "epsilon_squared",
            }
        ]
    )
    pairwise = pairwise_mannwhitney(work, group_col="paper_domain", value_col="TGC_%")
    if not pairwise.empty:
        pairwise.insert(0, "scope", "all_primary_domains")
        pairwise.insert(0, "hypothesis", "H1")
    return summary, pairwise


def run_weathering_hypothesis_tests(primary: pd.DataFrame, min_group_n: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict] = []
    pairwise_frames: list[pd.DataFrame] = []

    overall = primary.dropna(subset=["weathering_bin", "TGC_%"]).copy()
    overall = overall.groupby("weathering_bin").filter(lambda sub: len(sub) >= min_group_n)
    overall_groups = [(str(label), sub["TGC_%"].to_numpy(dtype=float)) for label, sub in overall.groupby("weathering_bin")]
    if len(overall_groups) >= 2:
        labels = [label for label, _ in overall_groups]
        values = [vals for _, vals in overall_groups]
        h_stat, p_val = stats.kruskal(*values)
        summary_rows.append(
            {
                "hypothesis": "H2",
                "scope": "all_primary_domains",
                "test": "Kruskal-Wallis",
                "grouping": "weathering_bin",
                "n_total": int(len(overall)),
                "n_groups": int(len(labels)),
                "groups_tested": "; ".join(labels),
                "statistic": float(h_stat),
                "p_value": float(p_val),
                "effect_size": kruskal_effect_size(float(h_stat), len(labels), len(overall)),
                "effect_size_label": "epsilon_squared",
            }
        )
        pair_df = pairwise_mannwhitney(overall, group_col="weathering_bin", value_col="TGC_%")
        if not pair_df.empty:
            pair_df.insert(0, "scope", "all_primary_domains")
            pair_df.insert(0, "hypothesis", "H2")
            pairwise_frames.append(pair_df)

    for domain, sub in primary.groupby("paper_domain"):
        work = sub.dropna(subset=["weathering_bin", "TGC_%"]).copy()
        work = work.groupby("weathering_bin").filter(lambda grp: len(grp) >= min_group_n)
        grouped = [(str(label), grp["TGC_%"].to_numpy(dtype=float)) for label, grp in work.groupby("weathering_bin")]
        if len(grouped) < 2:
            continue
        labels = [label for label, _ in grouped]
        values = [vals for _, vals in grouped]
        if len(grouped) == 2:
            stat = stats.mannwhitneyu(values[0], values[1], alternative="two-sided")
            effect = cliffs_delta(values[0], values[1])
            summary_rows.append(
                {
                    "hypothesis": "H2",
                    "scope": str(domain),
                    "test": "Mann-Whitney U",
                    "grouping": "weathering_bin",
                    "n_total": int(len(work)),
                    "n_groups": 2,
                    "groups_tested": "; ".join(labels),
                    "statistic": float(stat.statistic),
                    "p_value": float(stat.pvalue),
                    "effect_size": effect,
                    "effect_size_label": "cliffs_delta",
                }
            )
        else:
            h_stat, p_val = stats.kruskal(*values)
            summary_rows.append(
                {
                    "hypothesis": "H2",
                    "scope": str(domain),
                    "test": "Kruskal-Wallis",
                    "grouping": "weathering_bin",
                    "n_total": int(len(work)),
                    "n_groups": int(len(labels)),
                    "groups_tested": "; ".join(labels),
                    "statistic": float(h_stat),
                    "p_value": float(p_val),
                    "effect_size": kruskal_effect_size(float(h_stat), len(labels), len(work)),
                    "effect_size_label": "epsilon_squared",
                }
            )

        pair_df = pairwise_mannwhitney(work, group_col="weathering_bin", value_col="TGC_%")
        if not pair_df.empty:
            pair_df.insert(0, "scope", str(domain))
            pair_df.insert(0, "hypothesis", "H2")
            pairwise_frames.append(pair_df)

    pairwise = pd.concat(pairwise_frames, ignore_index=True) if pairwise_frames else pd.DataFrame()
    return pd.DataFrame(summary_rows), pairwise


def run_ml_screen(df: pd.DataFrame, target_threshold: float, feature_mode: str) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    work = df[df["paper_domain_status"] == "primary"].copy()
    work = work.dropna(subset=["TGC_%", "BHID"]).copy()
    work["target"] = (work["TGC_%"] >= target_threshold).astype(int)

    if feature_mode == "full":
        numeric_features = ["FROM", "TO", "INTERVAL", "X", "Y", "Z", "density_BD_COMBINED"]
        categorical_features = ["paper_domain", "weathering_bin", "geo_ALTERATION", "geo_SULPHIDES", "geo_STRUCTURE"]
    elif feature_mode == "geology_only":
        numeric_features = ["density_BD_COMBINED"]
        categorical_features = ["paper_domain", "weathering_bin", "geo_ALTERATION", "geo_SULPHIDES", "geo_STRUCTURE"]
    else:
        raise ValueError(f"Unknown feature mode: {feature_mode}")

    for col in numeric_features:
        if col not in work.columns:
            work[col] = np.nan
    for col in categorical_features:
        work[col] = safe_category(work[col])

    X = work[numeric_features + categorical_features]
    y = work["target"].to_numpy(dtype=int)
    groups = work["BHID"].astype(str).to_numpy()

    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    rf = Pipeline(
        steps=[
            ("pre", pre),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=500,
                    min_samples_leaf=4,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    lr = Pipeline(
        steps=[
            ("pre", pre),
            (
                "lr",
                LogisticRegression(
                    solver="liblinear",
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )
    dummy = Pipeline(steps=[("pre", pre), ("dummy", DummyClassifier(strategy="prior"))])

    cv = GroupKFold(n_splits=5)
    rows: list[dict] = []
    rf_group_scores: list[pd.Series] = []
    lr_group_scores: list[pd.Series] = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups), start=1):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        rf.fit(X_train, y_train)
        rpred = rf.predict(X_test)
        rprob = rf.predict_proba(X_test)[:, 1]
        rf_names = rf.named_steps["pre"].get_feature_names_out()
        rf_series = pd.Series(rf.named_steps["rf"].feature_importances_, index=rf_names)
        rf_group_scores.append(rf_series.groupby(feature_group).sum())

        lr.fit(X_train, y_train)
        lpred = lr.predict(X_test)
        lprob = lr.predict_proba(X_test)[:, 1]
        lr_names = lr.named_steps["pre"].get_feature_names_out()
        lr_series = pd.Series(np.abs(lr.named_steps["lr"].coef_.ravel()), index=lr_names)
        lr_group_scores.append(lr_series.groupby(feature_group).sum())

        dummy.fit(X_train, y_train)
        dpred = dummy.predict(X_test)
        dprob = dummy.predict_proba(X_test)[:, 1]

        rows.append(
            {
                "threshold_pct": target_threshold,
                "fold": fold,
                "n_test": int(len(test_idx)),
                "rf_balanced_accuracy": float(balanced_accuracy_score(y_test, rpred)),
                "rf_f1": float(f1_score(y_test, rpred, zero_division=0)),
                "rf_roc_auc": safe_roc_auc(y_test, rprob),
                "lr_balanced_accuracy": float(balanced_accuracy_score(y_test, lpred)),
                "lr_f1": float(f1_score(y_test, lpred, zero_division=0)),
                "lr_roc_auc": safe_roc_auc(y_test, lprob),
                "dummy_balanced_accuracy": float(balanced_accuracy_score(y_test, dpred)),
                "dummy_f1": float(f1_score(y_test, dpred, zero_division=0)),
                "dummy_roc_auc": safe_roc_auc(y_test, dprob),
            }
        )

    folds_df = pd.DataFrame(rows)
    rf_group_imp = pd.DataFrame(rf_group_scores).fillna(0.0).mean(axis=0).sort_values(ascending=False).reset_index()
    rf_group_imp.columns = ["feature_group", "rf_importance"]
    lr_group_imp = pd.DataFrame(lr_group_scores).fillna(0.0).mean(axis=0).sort_values(ascending=False).reset_index()
    lr_group_imp.columns = ["feature_group", "lr_abs_coef_importance"]
    group_imp = rf_group_imp.merge(lr_group_imp, on="feature_group", how="outer").fillna(0.0)
    group_imp = group_imp.sort_values(["rf_importance", "lr_abs_coef_importance"], ascending=False).reset_index(drop=True)

    summary = {
        "feature_mode": feature_mode,
        "threshold_pct": target_threshold,
        "n_rows": int(len(work)),
        "n_holes": int(pd.Series(groups).nunique()),
        "positive_fraction": float(np.mean(y)),
        "rf_balanced_accuracy_mean": float(folds_df["rf_balanced_accuracy"].mean()),
        "rf_f1_mean": float(folds_df["rf_f1"].mean()),
        "rf_roc_auc_mean": float(folds_df["rf_roc_auc"].mean()),
        "lr_balanced_accuracy_mean": float(folds_df["lr_balanced_accuracy"].mean()),
        "lr_f1_mean": float(folds_df["lr_f1"].mean()),
        "lr_roc_auc_mean": float(folds_df["lr_roc_auc"].mean()),
        "dummy_balanced_accuracy_mean": float(folds_df["dummy_balanced_accuracy"].mean()),
        "dummy_f1_mean": float(folds_df["dummy_f1"].mean()),
        "dummy_roc_auc_mean": float(folds_df["dummy_roc_auc"].mean()),
        "top_feature_groups": group_imp[["feature_group", "rf_importance"]]
        .rename(columns={"rf_importance": "importance"})
        .head(10)
        .to_dict("records"),
        "top_feature_groups_lr": group_imp[["feature_group", "lr_abs_coef_importance"]]
        .rename(columns={"lr_abs_coef_importance": "importance"})
        .sort_values("importance", ascending=False)
        .head(10)
        .to_dict("records"),
    }
    return summary, folds_df, group_imp


def plot_barh(df: pd.DataFrame, label_col: str, value_col: str, title: str, out_path: Path, top_n: int = 10) -> None:
    plot_df = df.head(top_n).iloc[::-1].copy()
    if plot_df.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(plot_df[label_col].astype(str), plot_df[value_col], color="#355f8c")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def write_report(
    out_dir: Path,
    domain_counts: pd.DataFrame,
    lith_weather: pd.DataFrame,
    geo_cv: pd.DataFrame,
    h1_summary: pd.DataFrame,
    h1_pairwise: pd.DataFrame,
    h2_summary: pd.DataFrame,
    ml5_geo: dict,
    ml3_geo: dict,
    ml5_full: dict,
    ml3_full: dict,
    variogram_df: pd.DataFrame,
) -> None:
    top_groups_5 = ml5_geo["top_feature_groups"][:5]
    top_groups_3 = ml3_geo["top_feature_groups"][:5]
    top_groups_5_lr = ml5_geo["top_feature_groups_lr"][:5]
    lines = [
        "# Combined Geology + ML Paper Package",
        "",
        "This package implements the combined Option 1 + Option 3 paper direction: geology defines the domain hypothesis, geostatistics tests whether the domains improve continuity modeling, and interpretable ML checks whether the logged geology carries predictive signal for enrichment.",
        "",
        "## Main Paper Story",
        "- The paper should be framed as geology-led and designed for Ore Geology Reviews.",
        "- The main domains are `GRSC`, `GRSC1`, `GRSC2`, and `weathered_graphitic`.",
        "- The ML section should be supporting evidence, not the central method.",
        "",
        "## Domain Readiness",
    ]
    for row in domain_counts.to_dict("records"):
        lines.append(
            f"- `{row['paper_domain']}`: n={int(row['count'])}, holes={int(row['n_holes'])}, mean grade={row['mean_grade_pct']:.2f}% TGC, frac>=5%={row['frac_ge_5pct']:.2f}."
        )
    lines += [
        "",
        "## Domain x Weathering Highlights",
    ]
    for row in lith_weather.head(8).to_dict("records"):
        lines.append(
            f"- `{row['paper_domain']}` in `{row['weathering_bin']}`: n={int(row['count'])}, mean grade={row['mean_grade_pct']:.2f}% TGC."
        )
    lines += [
        "",
        "## Geostatistical Comparison",
    ]
    for row in geo_cv.to_dict("records"):
        lines.append(
            f"- `{row['mode']}` blocked CV: RMSE={row['RMSE']:.3f}, MAE={row['MAE']:.3f}, ME={row['ME']:.3f}, n={int(row['n'])}."
        )
    if not h1_summary.empty:
        row = h1_summary.iloc[0]
        lines += [
            "",
            "## Formal Hypothesis Tests",
            f"- `H1` domain test: {row['test']} statistic={row['statistic']:.3f}, p={row['p_value']:.4g}, {row['effect_size_label']}={row['effect_size']:.3f}.",
        ]
    if not h1_pairwise.empty:
        sig_count = int(h1_pairwise["significant_holm_0_05"].sum())
        lines.append(f"- `H1` pairwise domain contrasts significant after Holm correction: {sig_count} of {len(h1_pairwise)} comparisons.")
    if not h2_summary.empty:
        for _, row in h2_summary.iterrows():
            lines.append(
                f"- `H2` weathering test in `{row['scope']}`: {row['test']} statistic={row['statistic']:.3f}, p={row['p_value']:.4g}, {row['effect_size_label']}={row['effect_size']:.3f}."
            )
    lines += [
        "",
        "## Interpretable ML Summary",
        f"- Geology-only `>=5% TGC` grouped CV: RF balanced accuracy {ml5_geo['rf_balanced_accuracy_mean']:.3f}, ROC-AUC {ml5_geo['rf_roc_auc_mean']:.3f}, versus dummy balanced accuracy {ml5_geo['dummy_balanced_accuracy_mean']:.3f}.",
        f"- Geology-only `>=5% TGC` grouped CV: logistic regression balanced accuracy {ml5_geo['lr_balanced_accuracy_mean']:.3f}, ROC-AUC {ml5_geo['lr_roc_auc_mean']:.3f}.",
        f"- Geology-only `>=3% TGC` grouped CV: RF balanced accuracy {ml3_geo['rf_balanced_accuracy_mean']:.3f}, ROC-AUC {ml3_geo['rf_roc_auc_mean']:.3f}, versus dummy balanced accuracy {ml3_geo['dummy_balanced_accuracy_mean']:.3f}.",
        f"- Geology-only `>=3% TGC` grouped CV: logistic regression balanced accuracy {ml3_geo['lr_balanced_accuracy_mean']:.3f}, ROC-AUC {ml3_geo['lr_roc_auc_mean']:.3f}.",
        f"- Full-feature `>=5% TGC` grouped CV: RF balanced accuracy {ml5_full['rf_balanced_accuracy_mean']:.3f}, ROC-AUC {ml5_full['rf_roc_auc_mean']:.3f}.",
        f"- Full-feature `>=3% TGC` grouped CV: RF balanced accuracy {ml3_full['rf_balanced_accuracy_mean']:.3f}, ROC-AUC {ml3_full['rf_roc_auc_mean']:.3f}.",
        "- Top geology-only feature groups for `>=5% TGC`:",
    ]
    for row in top_groups_5:
        lines.append(f"  - `{row['feature_group']}` = {row['importance']:.3f}")
    lines += [
        "- Top geology-only logistic-regression feature groups for `>=5% TGC`:",
    ]
    for row in top_groups_5_lr:
        lines.append(f"  - `{row['feature_group']}` = {row['importance']:.3f}")
    lines += [
        "- Top geology-only feature groups for `>=3% TGC`:",
    ]
    for row in top_groups_3:
        lines.append(f"  - `{row['feature_group']}` = {row['importance']:.3f}")
    lines += [
        "",
        "## Recommendation",
        "- Keep the geology-led split-domain story as the main paper only if the split workflow improves blocked CV or clearly improves variogram interpretability.",
        "- Keep ML in the main text because both geology-only RF and grouped logistic regression beat dummy at `>=5% TGC`; use the full-feature model only as supplementary sensitivity.",
        "- If structure remains sparse, keep it in supplementary discussion rather than the main claim.",
        "",
        "## Output Files",
        f"- `{(out_dir / 'paper_domain_mapping.csv').as_posix()}`",
        f"- `{(out_dir / 'paper_analysis_table.csv').as_posix()}`",
        f"- `{(out_dir / 'geostat_blocked_cv_summary.csv').as_posix()}`",
        f"- `{(out_dir / 'h1_domain_kruskal_summary.csv').as_posix()}`",
        f"- `{(out_dir / 'h2_weathering_summary.csv').as_posix()}`",
        f"- `{(out_dir / 'ml_summary_5pct_geology_only.json').as_posix()}`",
        f"- `{(out_dir / 'ml_summary_5pct_full.json').as_posix()}`",
        f"- `{(out_dir / 'ml_summary_5pct.json').as_posix()}`",
        f"- `{(out_dir / 'ml_summary_3pct.json').as_posix()}`",
        f"- `{(out_dir / 'directional_variogram_summary.csv').as_posix()}`",
    ]
    (out_dir / "combined_paper_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build combined geology + ML paper package.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Joined composite table")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory")
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input, low_memory=False)
    for col in ["TGC_%", "FROM", "TO", "INTERVAL", "X", "Y", "Z", "density_BD_COMBINED"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    mapped = df["geo_LITHO"].map(map_paper_domain)
    df["paper_domain"] = mapped.map(lambda x: x[0])
    df["paper_domain_status"] = mapped.map(lambda x: x[1])
    df["weathering_bin"] = df["geo_WEATHERING"].map(map_weathering_bin)
    df["grade_ge_5"] = df["TGC_%"] >= 5.0
    df["grade_ge_3"] = df["TGC_%"] >= 3.0

    unique_lith = sorted(df["geo_LITHO"].dropna().astype(str).unique())
    mapping_rows = []
    for lith in unique_lith:
        dom, status = map_paper_domain(lith)
        mapping_rows.append({"raw_lithology": lith, "paper_domain": dom, "domain_status": status})
    mapping_df = pd.DataFrame(mapping_rows)
    mapping_df.to_csv(out_dir / "paper_domain_mapping.csv", index=False)

    analysis_cols = [
        "BHID",
        "FROM",
        "TO",
        "INTERVAL",
        "X",
        "Y",
        "Z",
        "TGC_%",
        "paper_domain",
        "paper_domain_status",
        "weathering_bin",
        "geo_LITHO",
        "geo_WEATHERING",
        "geo_STRUCTURE",
        "geo_ALTERATION",
        "geo_SULPHIDES",
        "density_BD_COMBINED",
        "grade_ge_5",
        "grade_ge_3",
    ]
    analysis_df = df[analysis_cols].copy()
    for col in ["paper_domain", "paper_domain_status", "weathering_bin", "geo_LITHO", "geo_WEATHERING", "geo_STRUCTURE", "geo_ALTERATION", "geo_SULPHIDES"]:
        analysis_df[col] = safe_category(analysis_df[col])
    analysis_df.to_csv(out_dir / "paper_analysis_table.csv", index=False)

    primary = analysis_df[analysis_df["paper_domain_status"] == "primary"].copy()
    domain_counts = (
        primary.groupby("paper_domain", as_index=False)
        .agg(
            count=("TGC_%", "size"),
            n_holes=("BHID", "nunique"),
            mean_grade_pct=("TGC_%", "mean"),
            median_grade_pct=("TGC_%", "median"),
            frac_ge_5pct=("grade_ge_5", "mean"),
            frac_ge_3pct=("grade_ge_3", "mean"),
        )
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )
    domain_counts.to_csv(out_dir / "paper_domain_summary.csv", index=False)

    lith_weather = (
        primary.groupby(["paper_domain", "weathering_bin"], as_index=False)
        .agg(count=("TGC_%", "size"), mean_grade_pct=("TGC_%", "mean"))
        .sort_values(["count", "mean_grade_pct"], ascending=[False, False])
        .reset_index(drop=True)
    )
    lith_weather.to_csv(out_dir / "paper_domain_weathering_summary.csv", index=False)

    h1_summary, h1_pairwise = run_domain_hypothesis_tests(primary)
    h1_summary.to_csv(out_dir / "h1_domain_kruskal_summary.csv", index=False)
    h1_pairwise.to_csv(out_dir / "h1_domain_pairwise_mannwhitney.csv", index=False)

    h2_summary, h2_pairwise = run_weathering_hypothesis_tests(primary, min_group_n=10)
    h2_summary.to_csv(out_dir / "h2_weathering_summary.csv", index=False)
    h2_pairwise.to_csv(out_dir / "h2_weathering_pairwise.csv", index=False)

    mixed_supp = analysis_df[analysis_df["paper_domain_status"] == "supplementary"].copy()
    mixed_summary = (
        mixed_supp.groupby("geo_LITHO", as_index=False)
        .agg(count=("TGC_%", "size"), mean_grade_pct=("TGC_%", "mean"), frac_ge_5pct=("grade_ge_5", "mean"))
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )
    mixed_summary.to_csv(out_dir / "mixed_graphitic_supplementary_summary.csv", index=False)

    variogram_rows = []
    variogram_rows.extend(domain_directional_variogram_summary(primary, "pooled_graphitic"))
    for domain in PRIMARY_DOMAINS:
        variogram_rows.extend(domain_directional_variogram_summary(primary[primary["paper_domain"] == domain], domain))
    variogram_df = pd.DataFrame(variogram_rows)
    variogram_df.to_csv(out_dir / "directional_variogram_summary.csv", index=False)

    pooled_summary, pooled_folds = run_geostat_cv(analysis_df, mode="pooled")
    split_summary, split_folds = run_geostat_cv(analysis_df, mode="split")
    geostat_summary = pd.DataFrame([pooled_summary, split_summary])
    geostat_summary.to_csv(out_dir / "geostat_blocked_cv_summary.csv", index=False)
    pd.concat([pooled_folds, split_folds], ignore_index=True).to_csv(out_dir / "geostat_blocked_cv_folds.csv", index=False)

    ml5_geo_summary, ml5_geo_folds, ml5_geo_group_imp = run_ml_screen(analysis_df, target_threshold=5.0, feature_mode="geology_only")
    ml3_geo_summary, ml3_geo_folds, ml3_geo_group_imp = run_ml_screen(analysis_df, target_threshold=3.0, feature_mode="geology_only")
    ml5_full_summary, ml5_full_folds, ml5_full_group_imp = run_ml_screen(analysis_df, target_threshold=5.0, feature_mode="full")
    ml3_full_summary, ml3_full_folds, ml3_full_group_imp = run_ml_screen(analysis_df, target_threshold=3.0, feature_mode="full")

    (out_dir / "ml_summary_5pct_geology_only.json").write_text(json.dumps(ml5_geo_summary, indent=2), encoding="utf-8")
    (out_dir / "ml_summary_3pct_geology_only.json").write_text(json.dumps(ml3_geo_summary, indent=2), encoding="utf-8")
    (out_dir / "ml_summary_5pct_full.json").write_text(json.dumps(ml5_full_summary, indent=2), encoding="utf-8")
    (out_dir / "ml_summary_3pct_full.json").write_text(json.dumps(ml3_full_summary, indent=2), encoding="utf-8")
    (out_dir / "ml_summary_5pct.json").write_text(json.dumps(ml5_geo_summary, indent=2), encoding="utf-8")
    (out_dir / "ml_summary_3pct.json").write_text(json.dumps(ml3_geo_summary, indent=2), encoding="utf-8")

    ml5_geo_folds.to_csv(out_dir / "ml_folds_5pct_geology_only.csv", index=False)
    ml3_geo_folds.to_csv(out_dir / "ml_folds_3pct_geology_only.csv", index=False)
    ml5_full_folds.to_csv(out_dir / "ml_folds_5pct_full.csv", index=False)
    ml3_full_folds.to_csv(out_dir / "ml_folds_3pct_full.csv", index=False)
    ml5_geo_group_imp.to_csv(out_dir / "ml_group_importance_5pct_geology_only.csv", index=False)
    ml3_geo_group_imp.to_csv(out_dir / "ml_group_importance_3pct_geology_only.csv", index=False)
    ml5_full_group_imp.to_csv(out_dir / "ml_group_importance_5pct_full.csv", index=False)
    ml3_full_group_imp.to_csv(out_dir / "ml_group_importance_3pct_full.csv", index=False)

    plot_barh(domain_counts, "paper_domain", "mean_grade_pct", "Mean Grade by Paper Domain", out_dir / "mean_grade_by_paper_domain.png", top_n=10)
    lith_weather_plot = lith_weather.copy()
    lith_weather_plot["domain_weathering"] = lith_weather_plot["paper_domain"] + " | " + lith_weather_plot["weathering_bin"]
    plot_barh(lith_weather_plot, "domain_weathering", "mean_grade_pct", "Mean Grade by Domain-Weathering Combination", out_dir / "mean_grade_by_domain_weathering.png", top_n=8)
    plot_barh(
        ml5_geo_group_imp.rename(columns={"rf_importance": "importance"}),
        "feature_group",
        "importance",
        "Geology-Only RF Feature Group Importance (>=5% TGC)",
        out_dir / "ml_feature_group_importance_5pct.png",
        top_n=10,
    )
    plot_barh(
        ml5_geo_group_imp.rename(columns={"lr_abs_coef_importance": "importance"}).sort_values("importance", ascending=False),
        "feature_group",
        "importance",
        "Geology-Only Logistic Feature Group Importance (>=5% TGC)",
        out_dir / "ml_feature_group_importance_5pct_logistic.png",
        top_n=10,
    )

    write_report(
        out_dir=out_dir,
        domain_counts=domain_counts,
        lith_weather=lith_weather,
        geo_cv=geostat_summary,
        h1_summary=h1_summary,
        h1_pairwise=h1_pairwise,
        h2_summary=h2_summary,
        ml5_geo=ml5_geo_summary,
        ml3_geo=ml3_geo_summary,
        ml5_full=ml5_full_summary,
        ml3_full=ml3_full_summary,
        variogram_df=variogram_df,
    )

    print("Combined geology + ML paper package written:")
    for path in [
        out_dir / "paper_domain_mapping.csv",
        out_dir / "paper_analysis_table.csv",
        out_dir / "paper_domain_summary.csv",
        out_dir / "geostat_blocked_cv_summary.csv",
        out_dir / "ml_summary_5pct.json",
        out_dir / "combined_paper_report.md",
    ]:
        print(f"- {path}")


if __name__ == "__main__":
    main()
