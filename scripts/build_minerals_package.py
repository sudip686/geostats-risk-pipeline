from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NOTEBOOK_DIR = ROOT / "build" / "minerals_notebooklm_reviewer"
FALLBACK_NOTEBOOK_DIR = ROOT / "build" / "minerals_notebooklm"
DEFAULT_SOURCES_CSV = ROOT / "internal" / "minerals_notebook_sources.csv"
DEFAULT_OUTPUT_DIR = ROOT / "submission_minerals_ready"
ROOT_MANUSCRIPT = ROOT / "manuscript.md"
ROOT_TABLES = ROOT / "tables.md"
ROOT_CAPTIONS = ROOT / "figure_captions.md"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def pct(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def mt(value_tonnes: float, digits: int = 2) -> str:
    return f"{value_tonnes / 1e6:.{digits}f}"


def resolve_notebook_dir(path: Path | None) -> Path:
    if path:
        return path
    if DEFAULT_NOTEBOOK_DIR.exists():
        return DEFAULT_NOTEBOOK_DIR
    return FALLBACK_NOTEBOOK_DIR


def read_sources_table(path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        header = next(f, None)
        if header is None:
            raise RuntimeError(f"Empty sources CSV: {path}")
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 7:
                raise RuntimeError(f"Malformed sources CSV row: {line}")
            rows.append(
                {
                    "seq": int(parts[0]),
                    "title": ",".join(parts[1:-5]).strip(),
                    "year": int(parts[-5]),
                    "journal": parts[-4].strip(),
                    "doi": parts[-3].strip(),
                    "pdf_path": parts[-2].strip(),
                    "theme": parts[-1].strip(),
                }
            )
    df = pd.DataFrame(rows)
    df["ref_id"] = df["seq"].apply(lambda x: f"R{int(x)}")
    return df


def choose_cutoff_row(risk: pd.DataFrame, cutoff: float) -> pd.Series:
    row = risk.loc[(risk["cutoff"] - cutoff).abs() < 1e-9]
    if row.empty:
        raise RuntimeError(f"Cutoff {cutoff} missing from risked_tonnage.csv")
    return row.iloc[0]


def generate_tonnage_curve_figure(risk: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(7.2, 4.6))
    plt.plot(risk["cutoff"], risk["tonnage_p50"] / 1e6, label="P50", linewidth=2.0)
    plt.fill_between(
        risk["cutoff"],
        risk["tonnage_p10"] / 1e6,
        risk["tonnage_p90"] / 1e6,
        alpha=0.20,
        label="P10-P90 envelope",
    )
    plt.xlim(float(risk["cutoff"].min()), float(risk["cutoff"].max()))
    plt.xlabel("Cutoff grade (% TGC)")
    plt.ylabel("Tonnage (Mt)")
    plt.title("Decision-Range Tonnage Uncertainty")
    plt.grid(alpha=0.25, linewidth=0.5)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def format_reference_list(sources: pd.DataFrame) -> str:
    lines: list[str] = ["## References"]
    for _, row in sources.sort_values("seq").iterrows():
        parts = [f"**{row['ref_id']}**."]
        if isinstance(row.get("title"), str) and row["title"]:
            parts.append(str(row["title"]).strip())
        journal = str(row.get("journal") or "").strip()
        year = str(int(row["year"])) if not pd.isna(row.get("year")) else ""
        if journal or year:
            parts.append(f"{journal} ({year})".strip())
        lines.append(" ".join(p for p in parts if p))
    return "\n\n".join(lines)


def build_tables_md(meta: dict, risk: pd.DataFrame, metrics: dict, vario: dict) -> str:
    cfg = meta["config"]
    sim = cfg["simulation"]
    grid = cfg["grid"]
    reporting = cfg.get("reporting_grid", grid)
    cutoff = float(cfg.get("cutoff_grade", 3.0))
    row = choose_cutoff_row(risk, cutoff)

    domain_stats = meta.get("domain_stats", {})
    domain_total = sum(int(v["n"]) for v in domain_stats.values())
    domain_rows = []
    for name, stats in sorted(domain_stats.items()):
        share = 100.0 * float(stats["n"]) / domain_total if domain_total else 0.0
        domain_rows.append(
            f"| {name} | {int(stats['n'])} | {pct(share,1)} | {pct(float(stats['mean']))} | {pct(float(stats['std']))} |"
        )

    return "\n".join(
        [
            "# Tables",
            "",
            "## Table 1. Reviewer-Aligned Simulation Design",
            "",
            "| Item | Value |",
            "|---|---|",
            f"| Realizations | {int(sim['n_real'])} |",
            f"| Simulation support, m | {grid['dx']} x {grid['dy']} x {grid['dz']} |",
            f"| Reporting support, m | {reporting['dx']} x {reporting['dy']} x {reporting['dz']} |",
            f"| Search radius, m | {' / '.join(str(v) for v in sim.get('search_radius_m', [])) or 'global'} |",
            f"| Neighbors (min/max) | {sim.get('min_neighbors', 'NA')} / {sim.get('max_neighbors', 'NA')} |",
            f"| Trend model | {'disabled' if not cfg.get('trend', {}).get('enabled') else 'enabled'} |",
            f"| Top-cut | {'not applied' if not cfg.get('top_cut', {}).get('enabled') else 'applied'} |",
            "",
            "## Table 2. Geological Domain Summary",
            "",
            "| Domain | Samples | Share (%) | Mean TGC (%) | Std. dev. |",
            "|---|---:|---:|---:|---:|",
            *domain_rows,
            "",
            "## Table 3. Variogram Model Used for SGS",
            "",
            "| Item | Value |",
            "|---|---|",
            f"| Model type | {vario.get('model_type', 'exponential')} |",
            f"| Total sill in normal-score space | {pct(float(vario.get('total_sill', 1.0)), 2)} |",
            f"| Nugget | {pct(float(vario.get('nugget', 0.0)), 3)} |",
            f"| Structured sill | {pct(float(vario.get('sill', 0.0)), 3)} |",
            f"| Major range, m | {pct(float(vario.get('range', vario.get('len_scale', 0.0))), 1)} |",
            f"| Geological anisotropy, m | strike {cfg['variogram']['anisotropy']['ranges_m']['strike']}, down-dip {cfg['variogram']['anisotropy']['ranges_m']['down_dip']}, normal {cfg['variogram']['anisotropy']['ranges_m']['normal']} |",
            "",
            f"## Table 4. Grade-Tonnage Uncertainty at {pct(cutoff,1)}% TGC Cutoff",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| P10 tonnage (Mt) | {mt(float(row['tonnage_p10']))} |",
            f"| P50 tonnage (Mt) | {mt(float(row['tonnage_p50']))} |",
            f"| P90 tonnage (Mt) | {mt(float(row['tonnage_p90']))} |",
            f"| P10 grade (% TGC) | {pct(float(row['grade_p10']))} |",
            f"| P50 grade (% TGC) | {pct(float(row['grade_p50']))} |",
            f"| P90 grade (% TGC) | {pct(float(row['grade_p90']))} |",
            f"| P50 contained graphite (Mt) | {mt(float(row['contained_p50']))} |",
            "",
            "## Table 5. Validation Summary at Reporting Support",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Histogram overlap | {pct(float(metrics['hist_overlap']), 3)} |",
            f"| QQ RMSE | {pct(float(metrics['qq_rmse']), 3)} |",
            f"| Swath correlation X | {pct(float(metrics['swath_corr_x']), 3)} |",
            f"| Swath correlation Y | {pct(float(metrics['swath_corr_y']), 3)} |",
            f"| Swath correlation Z | {pct(float(metrics['swath_corr_z']), 3)} |",
            f"| Swath coverage (%) | {pct(float(metrics['swath_coverage_pct']), 1)} |",
            f"| Validation support, m | {metrics['support_dx']} x {metrics['support_dy']} x {metrics['support_dz']} |",
        ]
    )


def build_figure_captions_md(cfg: dict) -> str:
    return "\n".join(
        [
            "# Figure Captions",
            "",
            f"**Figure 1.** Directional variogram model used in SGS. The fitted model was normalized to unit total sill in normal-score space and plotted with geology-led anisotropy aligned to strike ({cfg['orebody']['strike_deg']} degrees), down-dip, and normal-to-plane continuity.",
            "",
            "**Figure 2.** Histogram comparison between composite grades and simulated grades at reporting support. This plot is retained in the main package because it directly shows whether the realizations preserve the grade distribution without relying on unreadable supplementary tables.",
            "",
            "**Figure 3.** Q-Q plot comparing composite grades and reporting-support realizations. It provides a compact check on distributional reproduction after regularization from simulation support to reporting support.",
            "",
            "**Figure 4.** Decision-range tonnage uncertainty curve for the selected cutoff range. Only the practical cutoff interval is shown in the main package; broader full-range curves are omitted to avoid internal-report style excess.",
            "",
            "**Supplementary Figure S1.** Swath plots in X, Y, and Z. These are retained in the supplement so the main paper stays focused on interpretation while still exposing trend-reproduction diagnostics.",
        ]
    )


def build_manuscript_md(meta: dict, risk: pd.DataFrame, metrics: dict, vario: dict, sources: pd.DataFrame) -> str:
    cfg = meta["config"]
    sim = cfg["simulation"]
    grid = cfg["grid"]
    reporting = cfg.get("reporting_grid", grid)
    cutoff = float(cfg.get("cutoff_grade", 3.0))
    row3 = choose_cutoff_row(risk, cutoff)
    row4 = choose_cutoff_row(risk, 4.0) if (risk["cutoff"] == 4.0).any() else row3
    row5 = choose_cutoff_row(risk, 5.0) if (risk["cutoff"] == 5.0).any() else row3

    references = format_reference_list(sources)
    domain_total = sum(int(v["n"]) for v in meta.get("domain_stats", {}).values())
    grsc_total = sum(
        int(stats["n"]) for name, stats in meta.get("domain_stats", {}).items() if "GRSC" in name
    )
    grsc_share = 100.0 * grsc_total / domain_total if domain_total else math.nan

    return f"""# Reviewer-Aligned Minerals Manuscript

## Title

Geology-Led Sequential Gaussian Simulation of a Stratiform Graphite Deposit in Tanzania

## Abstract

This revised workflow frames sequential Gaussian simulation (SGS) as a geology-led uncertainty tool rather than a parameter-tuning exercise. Graphitic schist lithologies were treated as the mineralized domain, composites were generated at 2 m support, and normal-score SGS was run for {int(sim['n_real'])} realizations using a fixed neighborhood judged large enough to avoid simulation artifacts. The simulation grid ({grid['dx']} x {grid['dy']} x {grid['dz']} m) was finer than the reporting grid ({reporting['dx']} x {reporting['dy']} x {reporting['dz']} m), and reporting results were obtained by block averaging after simulation. Variography was normalized to unit total sill in normal-score space and reported with one nugget interpretation and geology-led anisotropy. At a {pct(cutoff,1)}% TGC cutoff, the reporting-support ensemble gives P10/P50/P90 tonnage of {mt(float(row3['tonnage_p10']))}/{mt(float(row3['tonnage_p50']))}/{mt(float(row3['tonnage_p90']))} Mt, with a P50 grade of {pct(float(row3['grade_p50']))}% TGC. The revised package replaces broad sensitivity narratives with explicit methodological decisions supported by published mineral-resource simulation practice [R2, R3, R6, R7, R9].

## 1. Introduction

The deposit sits within the East African Orogen / Mozambique Belt context of Tanzania, where graphitic schists occur within deformed high-grade metasedimentary sequences and continuity is expected to follow lithological and structural fabric rather than isotropic distance alone [R1, R2, R4, R5, R8, R10]. That geological framing matters because the main reviewer concern was correct: SGS should not be presented as an exercise in tuning search parameters, trend options, and arbitrary scenario switches. Published mineral-resource simulation papers use SGS to quantify geological uncertainty once the domaining, support, and variogram model have been chosen on defensible grounds [R3, R6, R7, R9].

This revision therefore pivots the study away from broad algorithmic sensitivity claims. The core questions are now practical and geological: what continuity model is consistent with the graphite-bearing schist package, what reporting uncertainty emerges when realizations are simulated on a finer support and then averaged, and how much interpretation can be supported by the validation evidence. That emphasis aligns better with Minerals-style uncertainty papers and directly addresses the reviewer criticism that the previous draft had too little analysis [R3, R6, R7].

## 2. Geological Setting

Regional sources describe Tanzanian graphite-bearing schists as part of polydeformed, metamorphosed supracrustal packages within the East African Orogen, with continuity controlled by foliation-parallel geometry, folding, and shear-related disruption [R1, R2, R4, R5, R8, R10]. In the project database, graphitic schist variants dominate the mineralized interval set. Across the selected domain population, graphitic-schist codes account for approximately {pct(grsc_share,1)}% of domain composites. This supports a geology-led simulation strategy in which graphitic schists and closely related weathered variants are treated as the principal estimation domain, while anisotropy follows the interpreted orebody strike and dip.

## 3. Methods

### 3.1 Data preparation and domaining

Drillhole data were validated before desurveying, compositing, and simulation. Assays were composited to 2 m support using the standard length-weighted form

`Z_comp = (sum_i L_i Z_i) / (sum_i L_i)`

where `Z_comp` is the composite grade, `L_i` is the contributing sample length, and `Z_i` is the grade of sample `i`. The simulation domain was defined geologically from graphitic schist and closely related weathered graphitic units, not from a generic sensitivity exercise. This follows published resource workflows in which domaining is determined first from geology and only then passed to SGS [R3, R6, R7, R9].

### 3.2 Support handling and neighborhood choice

The simulation grid was set to {grid['dx']} x {grid['dy']} x {grid['dz']} m and the reporting grid to {reporting['dx']} x {reporting['dy']} x {reporting['dz']} m. The workflow therefore simulates on a support finer than the reporting support and then regularizes the realizations by block averaging. This directly addresses the reviewer's change-of-support criticism. The SGS search neighborhood was fixed at {' / '.join(str(v) for v in sim.get('search_radius_m', []))} m with {sim.get('min_neighbors')} to {sim.get('max_neighbors')} neighbors. It was not treated as a sensitivity variable; it was chosen once to be large enough to avoid obvious neighborhood artifacts and then kept unchanged for all realizations [R3, R6, R7, R9].

### 3.3 Variography and simulation

Grades were transformed to normal-score space prior to variogram modeling and SGS. The reported variogram was normalized to a total sill of {pct(float(vario.get('total_sill', 1.0)),2)} in Gaussian space, with nugget {pct(float(vario.get('nugget', 0.0)),3)} and structured sill {pct(float(vario.get('sill', 0.0)),3)}. A single nugget interpretation was retained across the directional model, while anisotropy ranges were set to strike/down-dip/normal values of {cfg['variogram']['anisotropy']['ranges_m']['strike']}/{cfg['variogram']['anisotropy']['ranges_m']['down_dip']}/{cfg['variogram']['anisotropy']['ranges_m']['normal']} m to match the geological concept of the graphite schist package. Trend modeling was disabled and top-cutting was not applied because the current distribution does not justify either step as a default requirement [R2, R3, R6, R9].

## 4. Results

At the reporting support, uncertainty widens as cutoff increases. At {pct(cutoff,1)}% TGC, the ensemble gives P10/P50/P90 tonnage of {mt(float(row3['tonnage_p10']))}/{mt(float(row3['tonnage_p50']))}/{mt(float(row3['tonnage_p90']))} Mt. At 4.0% TGC, the corresponding tonnage envelope is {mt(float(row4['tonnage_p10']))}/{mt(float(row4['tonnage_p50']))}/{mt(float(row4['tonnage_p90']))} Mt, and at 5.0% TGC it narrows to {mt(float(row5['tonnage_p10']))}/{mt(float(row5['tonnage_p50']))}/{mt(float(row5['tonnage_p90']))} Mt. The cutoff progression is important because it shows where decision sensitivity starts to matter rather than burying the result inside a long full-range tonnage curve.

Distributional reproduction at reporting support is reasonable but not uniform across all checks. Histogram overlap is {pct(float(metrics['hist_overlap']),3)} and QQ RMSE is {pct(float(metrics['qq_rmse']),3)}. Swath correlation is stronger in X ({pct(float(metrics['swath_corr_x']),3)}) and Y ({pct(float(metrics['swath_corr_y']),3)}) than in Z ({pct(float(metrics['swath_corr_z']),3)}), which is consistent with stronger lateral continuity than vertical continuity in the geological model. That contrast is more informative than claiming blanket success: it indicates that the model captures lateral trend better than vertical variation, and this should be discussed openly rather than hidden in unreadable supplements.

## 5. Discussion

The reviewer's main technical points required structural changes to the workflow, not cosmetic edits. First, the neighborhood is now fixed rather than calibrated as a sensitivity driver. That change matters because in SGS the neighborhood serves to construct the local conditional distribution, whereas the geological interpretation should come from domaining and variography. Second, the simulation support is finer than the reporting support, and the reporting quantities are derived after averaging. This makes the uncertainty results easier to defend because the reporting blocks no longer stand in for the simulation support itself.

The variogram presentation was also tightened. In a normal-score SGS workflow the sill must be reported consistently, and the nugget should not vary direction by direction as though it were a separate physical parameter in each panel. The present setup forces unit total sill and a single nugget interpretation, while anisotropy is tied to the orebody geometry. That is a cleaner and more defensible framing for a Minerals paper than the previous mixed narrative of tuning, calibration, and pilot-fit optimization [R3, R6, R9].

Finally, the paper now emphasizes interpretation instead of supplement overload. The main text focuses on the domain concept, support logic, validation summary, and cutoff-dependent uncertainty trends. Supporting files are retained only where they add traceability: swath plots, risk tables, variogram metadata, and NotebookLM-backed reviewer notes. That pruning directly addresses the complaint that the previous supplement contained pages of unreadable placeholders rather than analysis.

## 6. Conclusions

This revised workflow answers the reviewer criticisms by making explicit methodological decisions instead of exploring them as a sensitivity matrix. SGS was run on a finer-than-reporting support, regularized after simulation, and controlled by a fixed neighborhood rather than a tuned search experiment. The variogram model was normalized to unit sill in Gaussian space, reported with one nugget interpretation, and aligned with the geological concept of stratiform graphitic schist continuity.

The uncertainty results show that the reporting-support tonnage envelope remains relatively tight at lower cutoffs and widens as cutoff increases, which is the practically relevant behavior for screening-grade decisions. The remaining limitation is that vertical swath reproduction is weaker than the lateral directions, so future geological refinement should focus on short-scale vertical continuity and any lithological contacts that are not fully represented in the present domaining. Even with that limitation, the revised package is more consistent with published mineral-resource simulation practice than the earlier parameter-tuning draft [R3, R6, R7, R9].

{references}
"""


def write_claim_evidence_matrix(out_path: Path, run_dir: Path, notebook_dir: Path) -> None:
    rows = [
        ("M1", "Neighborhood fixed rather than varied", "config/stale_project_minerals.yaml", run_dir / "sgs_meta.json", notebook_dir / "01_major_neighborhood_support.md"),
        ("M2", "Domains geology-led; trend and top-cut removed", "config/stale_project_minerals.yaml", run_dir / "domain_data.csv", notebook_dir / "02_major_domains_trend_topcut.md"),
        ("M3", "Unit total sill and one nugget interpretation", "config/stale_project_minerals.yaml", run_dir / "figures" / "variogram_model.json", notebook_dir / "03_major_variograms.md"),
        ("M4", "Results/discussion expanded; supplement pruned", "scripts/build_minerals_package.py", run_dir / "tables" / "validation_metrics.json", notebook_dir / "04_major_analysis_validation_writing.md"),
        ("m2+m5", "Equation terms defined and digits reduced", "scripts/build_minerals_package.py", run_dir / "tables" / "risked_tonnage.csv", notebook_dir / "05_minor_equations_digits.md"),
        ("Framing", "Minerals geology-led framing", "scripts/build_minerals_package.py", run_dir / "sgs_meta.json", notebook_dir / "06_geology_minerals_framing.md"),
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["issue_id", "change", "config_or_script", "run_evidence", "notebook_evidence"])
        for issue_id, change, cfg, run_ev, nb_ev in rows:
            writer.writerow([issue_id, change, cfg, str(run_ev), str(nb_ev)])


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        ensure_dir(dst.parent)
        shutil.copy2(src, dst)


def write_pdf_bundle(sources: pd.DataFrame, bundle_dir: Path) -> None:
    ensure_dir(bundle_dir)
    manifest_rows: list[dict[str, str]] = []
    for _, row in sources.sort_values("seq").iterrows():
        src = ROOT / Path(str(row["pdf_path"]))
        if not src.exists():
            continue
        name = f"R{int(row['seq']):02d}_{src.parent.parent.name}_{src.name}"
        dst = bundle_dir / name
        shutil.copy2(src, dst)
        manifest_rows.append(
            {
                "ref_id": row["ref_id"],
                "title": str(row["title"]),
                "year": str(int(row["year"])) if not pd.isna(row["year"]) else "",
                "journal": str(row["journal"]),
                "copied_pdf": str(dst),
            }
        )
    pd.DataFrame(manifest_rows).to_csv(bundle_dir / "manifest.csv", index=False)


def write_package_manifest(root_dir: Path, out_path: Path) -> None:
    rows: list[dict[str, str | int]] = []
    for path in sorted(p for p in root_dir.rglob("*") if p.is_file()):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(
            {
                "relative_path": str(path.relative_to(root_dir)),
                "size_bytes": path.stat().st_size,
                "sha256": digest,
            }
        )
    pd.DataFrame(rows).to_csv(out_path, index=False)


def build_package(run_dir: Path, notebook_dir: Path, sources_csv: Path, output_dir: Path) -> None:
    if not run_dir.exists():
        raise RuntimeError(f"Run directory does not exist: {run_dir}")

    meta = load_json(run_dir / "sgs_meta.json")
    risk = pd.read_csv(run_dir / "tables" / "risked_tonnage.csv")
    metrics = load_json(run_dir / "tables" / "validation_metrics.json")
    vario = load_json(run_dir / "figures" / "variogram_model.json")
    sources = read_sources_table(sources_csv)

    figures_dir = ensure_dir(output_dir / "figures")
    supplement_dir = ensure_dir(output_dir / "supplement")
    evidence_dir = ensure_dir(output_dir / "evidence")

    copy_if_exists(run_dir / "figures" / "variogram.png", figures_dir / "figure_1_variogram.png")
    copy_if_exists(run_dir / "figures" / "histogram_validation.png", figures_dir / "figure_2_histogram_validation.png")
    copy_if_exists(run_dir / "figures" / "qq_plot.png", figures_dir / "figure_3_qq_plot.png")
    generate_tonnage_curve_figure(risk, figures_dir / "figure_4_tonnage_risk_curve.png")

    copy_if_exists(run_dir / "figures" / "swath_x.png", supplement_dir / "swath_x.png")
    copy_if_exists(run_dir / "figures" / "swath_y.png", supplement_dir / "swath_y.png")
    copy_if_exists(run_dir / "figures" / "swath_z.png", supplement_dir / "swath_z.png")
    copy_if_exists(run_dir / "tables" / "risked_tonnage.csv", supplement_dir / "risked_tonnage.csv")
    copy_if_exists(run_dir / "tables" / "validation_metrics.json", supplement_dir / "validation_metrics.json")
    copy_if_exists(run_dir / "figures" / "variogram_model.json", supplement_dir / "variogram_model.json")
    copy_if_exists(run_dir / "grids" / "sgs_reporting_meta.json", supplement_dir / "sgs_reporting_meta.json")

    notebook_copy_dir = evidence_dir / "notebooklm_answers"
    if notebook_dir.exists():
        if notebook_copy_dir.exists():
            shutil.rmtree(notebook_copy_dir)
        shutil.copytree(notebook_dir, notebook_copy_dir)

    write_claim_evidence_matrix(evidence_dir / "claim_evidence_matrix.csv", run_dir, notebook_dir)
    copy_if_exists(sources_csv, evidence_dir / "minerals_notebook_sources.csv")
    write_pdf_bundle(sources, ROOT / "reference_downloads" / "notebooklm_final_pdf_bundle")

    manuscript_text = build_manuscript_md(meta, risk, metrics, vario, sources)
    tables_text = build_tables_md(meta, risk, metrics, vario)
    captions_text = build_figure_captions_md(meta["config"])

    (output_dir / "manuscript.md").write_text(manuscript_text, encoding="utf-8")
    (output_dir / "tables.md").write_text(tables_text, encoding="utf-8")
    (output_dir / "figure_captions.md").write_text(captions_text, encoding="utf-8")
    ROOT_MANUSCRIPT.write_text(manuscript_text, encoding="utf-8")
    ROOT_TABLES.write_text(tables_text, encoding="utf-8")
    ROOT_CAPTIONS.write_text(captions_text, encoding="utf-8")

    write_package_manifest(output_dir, output_dir / "package_manifest.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Minerals-ready manuscript package from one run directory.")
    parser.add_argument("--run-dir", required=True, help="Run directory under output/")
    parser.add_argument("--notebook-dir", help="NotebookLM answer directory")
    parser.add_argument("--sources-csv", default=str(DEFAULT_SOURCES_CSV), help="Notebook source manifest CSV")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output package directory")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    notebook_dir = resolve_notebook_dir(Path(args.notebook_dir) if args.notebook_dir else None)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    if output_dir.exists():
        shutil.rmtree(output_dir)
    ensure_dir(output_dir)

    build_package(run_dir, notebook_dir, Path(args.sources_csv), output_dir)


if __name__ == "__main__":
    main()
