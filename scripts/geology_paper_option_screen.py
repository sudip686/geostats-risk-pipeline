from __future__ import annotations

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ZIP = ROOT / "internal" / "Tanga_MRE_2026-01-06 1 (1).zip"
DEFAULT_OUT_DIR = ROOT / "review" / "geology_first_paper_screen"

MODEL_MEMBER = "OneDrive_2026-01-06/Export Final/04 BM/CSV/MODEL_OK.csv"
ASSAY_MEMBER = "OneDrive_2026-01-06/Export Final/01 DB/CSV/DH_Assay.csv"
GEOLOGY_MEMBER = "OneDrive_2026-01-06/Export Final/01 DB/CSV/DH_Geology.csv"
COMPOSITE_MEMBER = "OneDrive_2026-01-06/Export Final/02 Composite/CSV/DH_Comp_2m.csv"
DENSITY_MEMBER = "OneDrive_2026-01-06/Export Final/01 DB/CSV/Density_All.csv"


def read_csv_from_zip(zip_path: Path, member: str) -> pd.DataFrame:
    with ZipFile(zip_path) as zf:
        with zf.open(member) as fh:
            return pd.read_csv(fh, low_memory=False)


def clean_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def overlap_len(a_from: float, a_to: float, b_from: float, b_to: float) -> float:
    return max(0.0, min(a_to, b_to) - max(a_from, b_from))


def attach_interval_table(
    base: pd.DataFrame,
    intervals: pd.DataFrame,
    keep_cols: list[str],
    prefix: str,
) -> pd.DataFrame:
    rows: list[dict] = []
    grouped = {str(k).strip(): g.copy() for k, g in intervals.groupby("BHID")}
    for rec in base.to_dict("records"):
        bhid = str(rec["BHID"]).strip()
        candidates = grouped.get(bhid)
        best: dict[str, object] = {}
        best_overlap = 0.0
        if candidates is not None:
            for _, item in candidates.iterrows():
                ov = overlap_len(float(rec["FROM"]), float(rec["TO"]), float(item["FROM"]), float(item["TO"]))
                if ov > best_overlap:
                    best_overlap = ov
                    best = {f"{prefix}_{col}": item.get(col) for col in keep_cols}
                    best[f"{prefix}_overlap_m"] = ov
        row = dict(rec)
        row.update(best)
        row[f"{prefix}_matched"] = best_overlap > 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_grade(df: pd.DataFrame, group_col: str, grade_col: str, min_count: int = 1) -> pd.DataFrame:
    sub = df[df[group_col].notna() & df[grade_col].notna()].copy()
    if sub.empty:
        return pd.DataFrame(columns=[group_col, "count", "mean", "median", "std", "frac_ge_3pct"])
    out = (
        sub.groupby(group_col)[grade_col]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
        .sort_values("count", ascending=False)
    )
    frac = (
        sub.assign(ge_3=sub[grade_col] >= 3.0)
        .groupby(group_col)["ge_3"]
        .mean()
        .reset_index(name="frac_ge_3pct")
    )
    out = out.merge(frac, on=group_col, how="left")
    return out[out["count"] >= min_count].reset_index(drop=True)


def plot_top_categories(df: pd.DataFrame, label_col: str, value_col: str, title: str, out_path: Path, top_n: int = 10) -> None:
    sub = df.head(top_n).iloc[::-1].copy()
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(sub[label_col].astype(str), sub[value_col], color="#365c8d")
    ax.set_title(title)
    ax.set_xlabel(value_col.replace("_", " "))
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def make_ml_screen(master_comp: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    work = master_comp.copy()
    target_col = "high_grade_ge_3pct"
    work[target_col] = work["TGC_%"] >= 3.0
    work = work[work["BHID"].notna() & work["TGC_%"].notna()].copy()

    numeric_features = ["FROM", "TO", "INTERVAL", "X", "Y", "Z", "density_BD_COMBINED"]
    categorical_features = [
        "geo_LITHO",
        "geo_WEATHERING",
        "geo_STRUCTURE",
        "geo_ALTERATION",
        "geo_SULPHIDES",
    ]

    for col in numeric_features:
        if col not in work.columns:
            work[col] = np.nan
    for col in categorical_features:
        if col not in work.columns:
            work[col] = np.nan

    X = work[numeric_features + categorical_features].copy()
    y = work[target_col].astype(int).to_numpy()
    groups = work["BHID"].astype(str).to_numpy()

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("pre", pre),
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=400,
                    min_samples_leaf=4,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    baseline = Pipeline(
        steps=[
            ("pre", pre),
            ("dummy", DummyClassifier(strategy="prior")),
        ]
    )

    n_groups = len(pd.unique(groups))
    n_splits = min(5, n_groups)
    if n_splits < 3:
        raise ValueError("Not enough drillholes for grouped CV.")
    cv = GroupKFold(n_splits=n_splits)

    rows: list[dict] = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups), start=1):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]

        baseline.fit(X_train, y_train)
        bpred = baseline.predict(X_test)
        bprob = baseline.predict_proba(X_test)[:, 1]

        rows.append(
            {
                "fold": fold,
                "n_test": int(len(test_idx)),
                "rf_balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
                "rf_f1": float(f1_score(y_test, pred, zero_division=0)),
                "rf_roc_auc": float(roc_auc_score(y_test, prob)),
                "dummy_balanced_accuracy": float(balanced_accuracy_score(y_test, bpred)),
                "dummy_f1": float(f1_score(y_test, bpred, zero_division=0)),
                "dummy_roc_auc": float(roc_auc_score(y_test, bprob)),
            }
        )

    fold_df = pd.DataFrame(rows)

    model.fit(X, y)
    pre_fitted = model.named_steps["pre"]
    rf_fitted = model.named_steps["rf"]
    feature_names = pre_fitted.get_feature_names_out()
    importances = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": rf_fitted.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    def group_feature(name: str) -> str:
        if name.startswith("num__"):
            return name.replace("num__", "", 1)
        if name.startswith("cat__"):
            body = name.replace("cat__", "", 1)
            return body.split("_", 1)[0]
        return name

    importances["feature_group"] = importances["feature"].map(group_feature)
    group_imp = (
        importances.groupby("feature_group", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    summary = {
        "n_rows": int(len(work)),
        "n_holes": int(n_groups),
        "positive_fraction": float(np.mean(y)),
        "cv_folds": int(n_splits),
        "rf_balanced_accuracy_mean": float(fold_df["rf_balanced_accuracy"].mean()),
        "rf_f1_mean": float(fold_df["rf_f1"].mean()),
        "rf_roc_auc_mean": float(fold_df["rf_roc_auc"].mean()),
        "dummy_balanced_accuracy_mean": float(fold_df["dummy_balanced_accuracy"].mean()),
        "dummy_f1_mean": float(fold_df["dummy_f1"].mean()),
        "dummy_roc_auc_mean": float(fold_df["dummy_roc_auc"].mean()),
        "top_feature_groups": group_imp.head(10).to_dict("records"),
        "top_encoded_features": importances.head(20).to_dict("records"),
    }
    return summary, fold_df, group_imp, importances


def read_metallurgy_summary() -> dict:
    purity = pd.read_excel(ROOT / "Purity and Recovery.xlsx")
    purity = purity[purity["Sample ID"].astype(str).str.match(r"^TDM\d+$", na=False)].copy()
    for col in ["Purity (TC Grade %)", "TC Recovery (%)"]:
        purity[col] = pd.to_numeric(purity[col], errors="coerce")

    flake_fresh = pd.read_excel(ROOT / "Flake Size Distribution_Fresh & Kaolinised Composites.xlsx")
    flake_oxide = pd.read_excel(ROOT / "Flake Size Distribution_Oxide Composites.xlsx")

    flake_stats: list[dict] = []
    for name, frame in [("fresh_kaolinised", flake_fresh), ("oxide_pair", flake_oxide)]:
        for sample in frame.columns[2:]:
            series = pd.to_numeric(frame[sample], errors="coerce")
            jumbo = float(series[frame["Market Category"] == "Jumbo"].sum())
            jumbo_large = float(series[frame["Market Category"].isin(["Jumbo", "Large"])].sum())
            flake_stats.append(
                {
                    "dataset": name,
                    "sample": sample,
                    "jumbo_pct": jumbo,
                    "jumbo_plus_large_pct": jumbo_large,
                }
            )

    return {
        "purity_recovery_rows": purity.to_dict("records"),
        "purity_recovery_means": purity.groupby("Oxidation State")[["Purity (TC Grade %)", "TC Recovery (%)"]]
        .mean()
        .reset_index()
        .to_dict("records"),
        "flake_size_summary": flake_stats,
    }


def build_markdown_report(
    out_dir: Path,
    coverage: dict,
    lith_summary: pd.DataFrame,
    weathering_summary: pd.DataFrame,
    structure_summary: pd.DataFrame,
    ml_summary: dict,
    metallurgy_summary: dict,
) -> str:
    top_lith = lith_summary.head(5).to_dict("records")
    top_weathering = weathering_summary.head(5).to_dict("records")
    top_structure = structure_summary.head(5).to_dict("records")
    top_groups = ml_summary["top_feature_groups"][:5]

    lines = [
        "# Geology-First Paper Option Screen",
        "",
        "This report is designed to give you paper styles that are clearly different from a standard mining-technology uncertainty paper. The common rule is: geology leads, geostatistics tests the geological idea, and machine learning only supports interpretation where useful.",
        "",
        "## Dataset Readiness",
        f"- Assay rows screened: {coverage['assay_rows']}",
        f"- Composite rows screened: {coverage['composite_rows']}",
        f"- Geology rows screened: {coverage['geology_rows']}",
        f"- Assay-geology match rate: {coverage['assay_geo_match_pct']:.1f}%",
        f"- Composite-geology match rate: {coverage['composite_geo_match_pct']:.1f}%",
        f"- Structure coverage in geology log: {coverage['structure_pct']:.1f}%",
        f"- Alteration coverage in geology log: {coverage['alteration_pct']:.1f}%",
        f"- Sulphide coverage in geology log: {coverage['sulphides_pct']:.1f}%",
        "",
        "## Key Geological Signals Already Present",
    ]

    for row in top_lith:
        lines.append(
            f"- Lithology `{row['geo_LITHO']}`: n={int(row['count'])}, mean grade={row['mean']:.2f}% TGC, frac>=3%={row['frac_ge_3pct']:.2f}."
        )
    for row in top_weathering:
        lines.append(
            f"- Weathering `{row['geo_WEATHERING']}`: n={int(row['count'])}, mean grade={row['mean']:.2f}% TGC, frac>=3%={row['frac_ge_3pct']:.2f}."
        )
    if top_structure:
        lines.append(
            f"- Structure logging exists but is sparse; best-covered structure class is `{top_structure[0]['geo_STRUCTURE']}` with n={int(top_structure[0]['count'])}."
        )
    else:
        lines.append("- Structure logging is too sparse in the joined composite table for a strong structure-led paper without relogging.")

    lines += [
        "",
        "## Paper Option 1: Weathering-Lithology Controls on Continuity",
        "**Style:** geology first, resource estimation second.",
        "**Hypothesis:** pooled graphitic domaining oversmooths real lithology- and weathering-controlled continuity differences.",
        "**Why this is different:** the paper is about geological controls on estimation behavior, not about uncertainty workflow alone.",
        "**How to prove it:**",
        "- Split graphitic units by lithology and weathering.",
        "- Compare domain-wise variograms, blocked CV, and swath behavior against the pooled model.",
        "- Use the joined composite table as the core evidence base.",
        "**How to disprove it:** if split domains do not improve blocked CV or produce stable variograms, stop.",
        "**Best fit:** Ore-geology or natural-resources journals.",
        "",
        "## Paper Option 2: Structural Geology Controls on Anisotropy",
        "**Style:** structural geology plus geostatistics.",
        "**Hypothesis:** the anisotropy used in estimation reflects real foliation or deformation architecture rather than just fitted variogram convenience.",
        "**Current feasibility:** moderate risk because structure coverage is only "
        f"{coverage['structure_pct']:.1f}% in the geology log.",
        "**How to prove it:**",
        "- Clean and relog structural fields from representative holes.",
        "- Build structural orientation groups and compare their anisotropy behavior.",
        "- Show that structure-conditioned anisotropy improves interpretability or predictive diagnostics.",
        "**How to disprove it:** if structure is too sparse or inconsistent to define stable regimes, drop this option.",
        "",
        "## Paper Option 3: Geology-Informed ML with Geostatistics in the Residuals",
        "**Style:** geology plus explainable ML plus geostatistics.",
        "**Hypothesis:** geology can predict high-grade graphite occurrence better than pooled coding alone, but spatial continuity still requires geostatistics.",
        f"**Current signal:** grouped drillhole CV gives RF balanced accuracy {ml_summary['rf_balanced_accuracy_mean']:.3f} and ROC-AUC {ml_summary['rf_roc_auc_mean']:.3f}, compared with dummy balanced accuracy {ml_summary['dummy_balanced_accuracy_mean']:.3f}.",
        "**Most important predictor groups in the current screen:**",
    ]
    for row in top_groups:
        lines.append(f"- `{row['feature_group']}` importance sum = {row['importance']:.3f}.")
    lines += [
        "**How to prove it:**",
        "- Keep grouped or spatial CV only; never use random CV as the main claim.",
        "- Show that the important predictors are geologically sensible.",
        "- Model residual spatial structure after ML, not instead of geostatistics.",
        "**How to disprove it:** if ML only works under leakage-prone validation and collapses under grouped CV, abandon this option.",
        "",
        "## Paper Option 4: Geometallurgy Linked Back to Geology",
        "**Style:** geology plus metallurgy, with optional ML later.",
        "**Hypothesis:** weathering and lithology predict recovery, purity, and flake-size behavior better than TGC alone.",
        "**Current signal:**",
    ]
    for row in metallurgy_summary["purity_recovery_means"]:
        lines.append(
            f"- `{row['Oxidation State']}` mean purity {row['Purity (TC Grade %)']:.2f}%, mean recovery {row['TC Recovery (%)']:.2f}%."
        )
    top_flake = sorted(metallurgy_summary["flake_size_summary"], key=lambda x: x["jumbo_plus_large_pct"], reverse=True)[:4]
    for row in top_flake:
        lines.append(
            f"- `{row['sample']}` jumbo+large share {row['jumbo_plus_large_pct']:.1f}%."
        )
    lines += [
        "**How to prove it:**",
        "- Link metallurgical composites back to lithology, weathering, and alteration.",
        "- Add targeted mineralogy or liberation work.",
        "- Test whether geology-aware models beat grade-only models.",
        "**How to disprove it:** if geology-aware models do not outperform TGC-only baselines, do not use this as the lead paper.",
        "",
        "## Recommendation",
        "Write **Option 1** first.",
        "Keep **Option 3** as the highest-upside alternative if you want a more modern geology-plus-ML paper.",
        "Do **Option 2** only if you are willing to improve structure logging.",
        "Use **Option 4** if you want a geology paper that naturally connects to metallurgy and product quality rather than another estimation paper.",
        "",
        "## Output Files",
        f"- Joined assay table: `{(out_dir / 'master_assay_geology_density.csv').as_posix()}`",
        f"- Joined composite table: `{(out_dir / 'master_composite_geology_density.csv').as_posix()}`",
        f"- Lithology summary: `{(out_dir / 'grade_by_lithology.csv').as_posix()}`",
        f"- Weathering summary: `{(out_dir / 'grade_by_weathering.csv').as_posix()}`",
        f"- ML summary: `{(out_dir / 'ml_high_grade_screen_summary.json').as_posix()}`",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate geology-first paper options from Tanga data.")
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP, help="ZIP file containing model and DB exports")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output directory")
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    assay = clean_numeric(read_csv_from_zip(args.zip, ASSAY_MEMBER), ["FROM", "TO", "INTERVAL", "Graphitic Carbon (TGC%)", "X", "Y", "Z"])
    geology = clean_numeric(read_csv_from_zip(args.zip, GEOLOGY_MEMBER), ["FROM", "TO", "INTERVAL"])
    composite = clean_numeric(read_csv_from_zip(args.zip, COMPOSITE_MEMBER), ["FROM", "TO", "INTERVAL", "TGC_%", "X", "Y", "Z"])
    density = clean_numeric(read_csv_from_zip(args.zip, DENSITY_MEMBER), ["FROM", "TO", "BD_COMBINED"])
    model = clean_numeric(read_csv_from_zip(args.zip, MODEL_MEMBER), ["X", "_X", "Y", "_Y", "Z", "_Z", "TGC_%", "DENSITY"])

    for frame in [assay, geology, composite, density]:
        frame["BHID"] = frame["BHID"].astype(str).str.strip()

    assay = assay.rename(columns={"Graphitic Carbon (TGC%)": "TGC_%"}).dropna(subset=["BHID", "FROM", "TO", "TGC_%"]).copy()
    composite = composite.dropna(subset=["BHID", "FROM", "TO", "TGC_%"]).copy()
    geology = geology.dropna(subset=["BHID", "FROM", "TO", "LITHO"]).copy()
    density = density.dropna(subset=["BHID", "FROM", "TO"]).copy()

    geo_keep = ["LITHO", "WEATHERING", "STRUCTURE", "ALTERATION", "SULPHIDES", "TYPE OF STRUCTURE (S0/S1/S2/L1/L2)"]
    dens_keep = ["BD_COMBINED", "OXIDE", "TYPE_COMBO"]

    master_assay = attach_interval_table(assay, geology, geo_keep, prefix="geo")
    master_assay = attach_interval_table(master_assay, density, dens_keep, prefix="density")

    master_comp = attach_interval_table(composite, geology, geo_keep, prefix="geo")
    master_comp = attach_interval_table(master_comp, density, dens_keep, prefix="density")

    master_assay.to_csv(out_dir / "master_assay_geology_density.csv", index=False)
    master_comp.to_csv(out_dir / "master_composite_geology_density.csv", index=False)

    geology_lengths = geology.copy()
    geology_lengths["length_m"] = geology_lengths["TO"] - geology_lengths["FROM"]
    lith_length = (
        geology_lengths.groupby("LITHO", as_index=False)["length_m"]
        .sum()
        .sort_values("length_m", ascending=False)
        .reset_index(drop=True)
    )
    lith_length["pct_of_logged_length"] = lith_length["length_m"] / lith_length["length_m"].sum() * 100.0
    lith_length.to_csv(out_dir / "lithology_logged_length_summary.csv", index=False)

    grade_lith = summarize_grade(master_comp, "geo_LITHO", "TGC_%", min_count=5)
    grade_weathering = summarize_grade(master_comp, "geo_WEATHERING", "TGC_%", min_count=5)
    grade_structure = summarize_grade(master_comp, "geo_STRUCTURE", "TGC_%", min_count=3)
    grade_alteration = summarize_grade(master_comp, "geo_ALTERATION", "TGC_%", min_count=5)
    grade_sulphides = summarize_grade(master_comp, "geo_SULPHIDES", "TGC_%", min_count=5)

    grade_lith.to_csv(out_dir / "grade_by_lithology.csv", index=False)
    grade_weathering.to_csv(out_dir / "grade_by_weathering.csv", index=False)
    grade_structure.to_csv(out_dir / "grade_by_structure.csv", index=False)
    grade_alteration.to_csv(out_dir / "grade_by_alteration.csv", index=False)
    grade_sulphides.to_csv(out_dir / "grade_by_sulphides.csv", index=False)

    plot_top_categories(grade_lith, "geo_LITHO", "mean", "Mean Composite Grade by Lithology", out_dir / "mean_grade_by_lithology.png")
    plot_top_categories(grade_weathering, "geo_WEATHERING", "mean", "Mean Composite Grade by Weathering", out_dir / "mean_grade_by_weathering.png")
    plot_top_categories(lith_length, "LITHO", "length_m", "Logged Length by Lithology", out_dir / "logged_length_by_lithology.png")

    ml_summary, ml_fold_df, ml_group_imp, ml_feature_imp = make_ml_screen(master_comp)
    ml_fold_df.to_csv(out_dir / "ml_high_grade_fold_metrics.csv", index=False)
    ml_group_imp.to_csv(out_dir / "ml_high_grade_group_importance.csv", index=False)
    ml_feature_imp.to_csv(out_dir / "ml_high_grade_feature_importance.csv", index=False)
    (out_dir / "ml_high_grade_screen_summary.json").write_text(json.dumps(ml_summary, indent=2), encoding="utf-8")
    plot_top_categories(ml_group_imp, "feature_group", "importance", "Geology-Informed ML: Feature Group Importance", out_dir / "ml_group_importance.png", top_n=8)

    metallurgy_summary = read_metallurgy_summary()
    (out_dir / "metallurgy_screen_summary.json").write_text(json.dumps(metallurgy_summary, indent=2), encoding="utf-8")

    coverage = {
        "assay_rows": int(len(assay)),
        "composite_rows": int(len(composite)),
        "geology_rows": int(len(geology)),
        "density_rows": int(len(density)),
        "model_rows": int(model.dropna(subset=["TGC_%"]).shape[0]),
        "assay_geo_match_pct": float(master_assay["geo_matched"].mean() * 100.0),
        "composite_geo_match_pct": float(master_comp["geo_matched"].mean() * 100.0),
        "assay_density_match_pct": float(master_assay["density_matched"].mean() * 100.0),
        "composite_density_match_pct": float(master_comp["density_matched"].mean() * 100.0),
        "structure_pct": float(geology["STRUCTURE"].notna().mean() * 100.0),
        "alteration_pct": float(geology["ALTERATION"].notna().mean() * 100.0),
        "sulphides_pct": float(geology["SULPHIDES"].notna().mean() * 100.0),
    }
    (out_dir / "data_coverage_summary.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")

    report = build_markdown_report(
        out_dir=out_dir,
        coverage=coverage,
        lith_summary=grade_lith,
        weathering_summary=grade_weathering,
        structure_summary=grade_structure,
        ml_summary=ml_summary,
        metallurgy_summary=metallurgy_summary,
    )
    report_path = out_dir / "geology_first_paper_options.md"
    report_path.write_text(report, encoding="utf-8")

    print("Geology-first paper screen outputs written:")
    for path in [
        report_path,
        out_dir / "master_composite_geology_density.csv",
        out_dir / "grade_by_lithology.csv",
        out_dir / "grade_by_weathering.csv",
        out_dir / "ml_high_grade_screen_summary.json",
    ]:
        print(f"- {path}")


if __name__ == "__main__":
    main()
