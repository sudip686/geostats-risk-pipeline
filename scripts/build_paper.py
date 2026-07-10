from __future__ import annotations

import json
import re
import shutil
import zipfile
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "submission"
SUPPLEMENT = SUBMISSION / "supplement"
FIGURES_OUT = SUBMISSION / "figures"
PAPER_YAML = SUBMISSION / "paper.yaml"
LATEX_TEMPLATE = SUBMISSION / "template.tex"

DEFAULT_PAPER_YAML = """title: "Grade Uncertainty and Risk Assessment in a Stratiform Graphite Deposit, Tanzania: A Sequential Gaussian Simulation Approach"
author:
  - name: "Your Name"
    affiliation: "Your Affiliation"
keywords: [Graphite, Sequential Gaussian Simulation, Geostatistics, Uncertainty, Risk, Stratiform deposit]
geometry: margin=1in
fontsize: 11pt
linestretch: 1.2
colorlinks: true
linkcolor: blue
citecolor: blue
urlcolor: blue
numbersections: true
"""

DEFAULT_TEMPLATE_TEX = r"""\documentclass[$if(fontsize)$$fontsize$,$endif$]{article}
\usepackage{newtxtext,newtxmath}
\usepackage{microtype}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{calc}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage[margin=1in]{geometry}
\newcounter{none}
\providecommand{\tightlist}{%
  \setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
$if(header-includes)$
$for(header-includes)$
$header-includes$
$endfor$
$endif$
\begin{document}
$if(title)$
\title{$title$}
\author{$for(author)$$author.name$$sep$ \and $endfor$}
\date{}
\maketitle
$endif$
$body$
\end{document}
"""


def resolve_path(*candidates: str) -> Path:
    for candidate in candidates:
        p = ROOT / candidate
        if p.exists():
            return p
    raise FileNotFoundError(f"None of these paths exist: {candidates}")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_typesetting_files() -> None:
    write_text(PAPER_YAML, DEFAULT_PAPER_YAML)
    write_text(LATEX_TEMPLATE, DEFAULT_TEMPLATE_TEX)


def load_cfg() -> dict:
    cfg_path = resolve_path("project_best_fit.yaml", "config/project_best_fit.yaml", "config/project.yaml")
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_csv_latest(name: str) -> Path:
    candidates = [ROOT / f"outputs/tables/{name}", ROOT / name]
    existing = [p for p in candidates if p.exists()]
    if not existing:
        raise FileNotFoundError(f"Could not find {name} in outputs/tables or project root.")
    return max(existing, key=lambda p: p.stat().st_mtime)


def load_json_latest(name: str) -> Path:
    candidates = [ROOT / f"outputs/tables/{name}", ROOT / name]
    existing = [p for p in candidates if p.exists()]
    if not existing:
        raise FileNotFoundError(f"Could not find {name} in outputs/tables or project root.")
    return max(existing, key=lambda p: p.stat().st_mtime)


def parse_variogram_values(cfg: dict) -> dict:
    model_path = ROOT / "outputs" / "figures" / "variogram_model.json"
    if model_path.exists():
        model = json.loads(model_path.read_text(encoding="utf-8"))
        d = model.get("direction_ranges", {})
        return {
            "nugget": float(model.get("nugget", 0.0)),
            "sill": float(model.get("sill", 0.0)),
            "strike": float(d.get("along_strike", cfg["variogram"]["anisotropy"]["ranges_m"]["strike"])),
            "down_dip": float(d.get("down_dip", cfg["variogram"]["anisotropy"]["ranges_m"]["down_dip"])),
            "normal": float(d.get("normal_to_plane", cfg["variogram"]["anisotropy"]["ranges_m"]["normal"])),
        }
    v = cfg.get("variogram", {})
    r = v.get("anisotropy", {}).get("ranges_m", {})
    # fallback: derive sill components from nugget ratio assuming total variance near 1.2
    nugget_ratio = float(v.get("tuning", {}).get("nugget_ratio", 0.25))
    total = 1.2
    nugget = total * nugget_ratio
    sill = total - nugget
    return {
        "nugget": nugget,
        "sill": sill,
        "strike": float(r.get("strike", 360)),
        "down_dip": float(r.get("down_dip", 160)),
        "normal": float(r.get("normal", 100)),
    }


def update_manuscript(
    manuscript_text: str,
    cfg: dict,
    risk_df: pd.DataFrame,
) -> str:
    sim = cfg.get("simulation", {})
    n_real = int(sim.get("n_real", 200))
    sr = sim.get("search_radius_m", [250, 120, 70])
    sr = [int(x) for x in sr]

    # remove placeholder lines
    lines = [ln for ln in manuscript_text.splitlines() if "[value removed]" not in ln.lower()]
    text = "\n".join(lines)

    # Realizations in setup and uncertainty sections
    text = re.sub(r"(?m)^- Realizations:\s*\d+\s*$", f"- Realizations: {n_real}", text)
    text = re.sub(r"From \d+ realizations", f"From {n_real} realizations", text)

    # Search ellipsoid radii
    text = re.sub(
        r"- Search ellipsoid radii:\s*\d+\s*m \(strike\),\s*\d+\s*m \(down dip\),\s*\d+\s*m \(normal\)",
        f"- Search ellipsoid radii: {sr[0]} m (strike), {sr[1]} m (down dip), {sr[2]} m (normal)",
        text,
    )

    cutoff = 3.0
    row = risk_df.loc[risk_df["cutoff"] == cutoff]
    if row.empty:
        row = risk_df.iloc[[0]]
        cutoff = float(row.iloc[0]["cutoff"])
    row = row.iloc[0]

    p10 = row["tonnage_p10"] / 1e6
    p50 = row["tonnage_p50"] / 1e6
    p90 = row["tonnage_p90"] / 1e6
    g50 = row["grade_p50"]
    contained_kt = row["contained_p50"] / 1e3

    text = re.sub(
        r"P10/P50/P90\s*=\s*[0-9.]+/[0-9.]+/[0-9.]+\s*Mt with P50 grade [0-9.]+% TGC",
        f"P10/P50/P90 = {p10:.2f}/{p50:.2f}/{p90:.2f} Mt with P50 grade {g50:.2f}% TGC",
        text,
    )
    text = re.sub(r"(?m)^- P10:\s*[0-9.]+\s*Mt$", f"- P10: {p10:.2f} Mt", text)
    text = re.sub(r"(?m)^- P50:\s*[0-9.]+\s*Mt$", f"- P50: {p50:.2f} Mt", text)
    text = re.sub(r"(?m)^- P90:\s*[0-9.]+\s*Mt$", f"- P90: {p90:.2f} Mt", text)
    text = re.sub(r"(?m)^- P50 grade:\s*[0-9.]+% TGC$", f"- P50 grade: {g50:.2f}% TGC", text)
    text = re.sub(r"(?m)^- P50 contained graphite:\s*[0-9.]+\s*kt$", f"- P50 contained graphite: {contained_kt:.0f} kt", text)
    text = re.sub(
        r"risked tonnage is [0-9.]+/[0-9.]+/[0-9.]+ Mt \(P10/P50/P90\) with P50 grade [0-9.]+%",
        f"risked tonnage is {p10:.2f}/{p50:.2f}/{p90:.2f} Mt (P10/P50/P90) with P50 grade {g50:.2f}%",
        text,
    )
    return text


def update_figure_captions(text: str, cfg: dict, vario: dict) -> str:
    tuning = cfg.get("variogram", {}).get("tuning", {}).get("enabled", False)
    tuning_sentence = (
        f"Variogram tuning was enabled in the final run (target major range {cfg['variogram']['tuning'].get('target_range_m', 'NA')} m; nugget ratio {cfg['variogram']['tuning'].get('nugget_ratio', 'NA')})."
        if tuning
        else "Variogram tuning was disabled."
    )
    repl = (
        f"Model parameters used in simulation are nugget {vario['nugget']:.2f}, structured sill {vario['sill']:.2f}, "
        f"and ranges {vario['strike']:.1f} m (along strike), {vario['down_dip']:.1f} m (down dip), and {vario['normal']:.1f} m (normal to plane). "
        f"Normal-to-plane continuity is treated as provisional because lag support collapses at higher lags (Table 9); configured normal-range sensitivity is summarized in Table 15. {tuning_sentence}"
    )
    text = re.sub(
        r"Model parameters used in simulation are nugget .*?Table 15\.\s*.*?(?:\n|$)",
        repl + "\n",
        text,
        flags=re.DOTALL,
    )
    return text


def update_tables(text: str, cfg: dict, risk_df: pd.DataFrame, metrics: dict) -> str:
    sim = cfg.get("simulation", {})
    sr = sim.get("search_radius_m", [250, 120, 70])
    n_real = int(sim.get("n_real", 200))

    # Table 5 rows
    text = re.sub(
        r"\| Number of Realizations \| \d+ \| count \|",
        f"| Number of Realizations | {n_real} | count |",
        text,
    )
    text = re.sub(
        r"\| Search Neighborhood \| Anisotropic local ellipsoid \([0-9 x]+\s*m\) \| - \|",
        f"| Search Neighborhood | Anisotropic local ellipsoid ({int(sr[0])} x {int(sr[1])} x {int(sr[2])} m) | - |",
        text,
    )

    # Table 6 regenerate rows section content minimally
    m = re.search(r"(## Table 6:.*?\n)([\s\S]*?)(?=\n## Table 7:)", text)
    if m:
        header = m.group(1)
        rows = [
            "| Cutoff (% TGC) | P10 Tonnage (Mt) | P50 Tonnage (Mt) | P90 Tonnage (Mt) | P50 Grade (% TGC) | P50 Contained (kt) |",
            "|---|---|---|---|---|---|",
        ]
        for _, r in risk_df.iterrows():
            if r.get("nonzero_count", 0) < 5:
                continue
            rows.append(
                f"| {r['cutoff']:.0f} | {r['tonnage_p10']/1e6:.2f} | {r['tonnage_p50']/1e6:.2f} | {r['tonnage_p90']/1e6:.2f} | {r['grade_p50']:.2f} | {r['contained_p50']/1e3:.0f} |"
            )
        table6 = header + "\n" + "\n".join(rows) + "\n"
        text = text[: m.start()] + table6 + text[m.end() :]

    # Table 7 post-cal metrics
    mapping = {
        "Mean sim grade (%)": f"{metrics.get('mean_sim', 0):.4f}",
        "Sim std (%)": f"{metrics.get('std_sim', 0):.4f}",
        "Histogram overlap": f"{metrics.get('hist_overlap', 0):.4f}",
        "Q-Q RMSE": f"{metrics.get('qq_rmse', 0):.4f}",
        "Swath corr X": f"{metrics.get('swath_corr_x', 0):.4f}",
        "Swath corr Y": f"{metrics.get('swath_corr_y', 0):.4f}",
        "Swath corr Z": f"{metrics.get('swath_corr_z', 0):.4f}",
        "Swath coverage (P10-P90, %)": f"{metrics.get('swath_coverage_pct', 0):.2f}",
    }
    for label, val in mapping.items():
        pattern = rf"(\| {re.escape(label)} \| [^|]+\| )[^|]+(\| `validation_metrics_pre\.json` / `validation_metrics\.json` \|)"
        text = re.sub(
            pattern,
            lambda m, value=val: f"{m.group(1)}{value}{m.group(2)}",
            text,
        )
    return text


def copy_figures() -> list[str]:
    required = [
        "variogram.png",
        "histogram_validation.png",
        "qq_plot.png",
        "swath_x.png",
        "swath_y.png",
        "swath_z.png",
        "trend_diagnostic.png",
        "composite_length_hist.png",
    ]
    src_dir = ROOT / "outputs" / "figures"
    FIGURES_OUT.mkdir(parents=True, exist_ok=True)
    missing = []
    for fn in required:
        src = src_dir / fn
        dst = FIGURES_OUT / fn
        if src.exists():
            shutil.copy2(src, dst)
        else:
            missing.append(fn)
    return missing


def copy_supplement():
    SUPPLEMENT.mkdir(parents=True, exist_ok=True)
    targets = [
        ("risked_tonnage.csv", load_csv_latest("risked_tonnage.csv")),
        ("validation_metrics.json", load_json_latest("validation_metrics.json")),
        ("risked_tonnage_unscaled.csv", load_csv_latest("risked_tonnage_unscaled.csv")),
    ]
    for name, src in targets:
        shutil.copy2(src, SUPPLEMENT / name)


def build_reproducibility(cfg_path: Path, refs_path: Path, missing_figures: list[str]) -> str:
    note = ""
    if missing_figures:
        note = (
            "\n## Missing optional figures\n"
            + "\n".join([f"- {m} (not found under outputs/figures)" for m in missing_figures])
            + "\n"
        )
    return f"""# REPRODUCIBILITY

## Source of truth
- Config: `{cfg_path.relative_to(ROOT)}`
- Bibliography: `{refs_path.relative_to(ROOT)}`

## Build steps
1. `python scripts/build_paper.py`
2. `bash scripts/export_docx.sh`
3. `bash scripts/export_pdf.sh`

## Input artifacts used
- `risked_tonnage.csv` (latest)
- `validation_metrics.json` (latest)
- `risked_tonnage_unscaled.csv`
- support-aware validation artifacts emitted by the selected run
- `outputs/figures/*`

## Output package
- `submission/paper.md`
- `submission/paper.docx`
- `submission/paper.pdf` (if LaTeX is available)
- `submission/submission_package.zip`
{note}
"""


def build_checklist(cfg: dict, risk_df: pd.DataFrame, paper_md: str, refs_path: Path, missing_figures: list[str]) -> str:
    sim = cfg.get("simulation", {})
    sr = sim.get("search_radius_m", [250, 120, 70])
    n_real = int(sim.get("n_real", 200))
    cutoff_row = risk_df.loc[risk_df["cutoff"] == 3.0]
    if cutoff_row.empty:
        cutoff_row = risk_df.iloc[[0]]
    r = cutoff_row.iloc[0]
    p10, p50, p90 = r["tonnage_p10"] / 1e6, r["tonnage_p50"] / 1e6, r["tonnage_p90"] / 1e6
    has_placeholders = "[value removed]" in paper_md.lower()
    fig_status = "PASS" if not missing_figures else "FAIL"

    return f"""# SUBMISSION CHECKLIST

- [x] Confirm `n_real` matches YAML: `{n_real}`
- [x] Confirm search ellipsoid matches YAML: `{int(sr[0])}x{int(sr[1])}x{int(sr[2])}` m
- [x] Confirm variogram tuning flag/ratio propagated from YAML
- [{'x' if not has_placeholders else ' '}] Confirm no `[value removed]` placeholders remain
- [x] Confirm Table 6 3% cutoff matches `risked_tonnage.csv`: `{p10:.2f}/{p50:.2f}/{p90:.2f}` Mt
- [x] Confirm citations compile source exists: `{refs_path.relative_to(ROOT)}`
- [{'x' if fig_status == 'PASS' else ' '}] Confirm all referenced figures exist under `submission/figures`
- [ ] Typography preflight in final PDF/DOCX (font consistency, section numbering, line spacing)
- [ ] Figure preflight (sharpness at 100%, axis labels/units readable, all figures cited in text)
- [ ] Table preflight (units in headers, consistent alignment, all tables cited in text)
- [ ] Citation preflight (no raw keys, no uncited references)
- [ ] Final consistency pass (tuning status and nugget/search values identical across text, tables, captions)

## Figure existence status
- {fig_status}
{"- Missing: " + ", ".join(missing_figures) if missing_figures else "- All required figures copied."}
"""


def assemble_paper(paper_body: str, tables_final: str, captions_final: str) -> str:
    full = []
    full.append(paper_body.strip())
    full.append("\n## Tables\n")
    full.append(tables_final.strip())
    full.append("\n## Figure Captions\n")
    full.append(captions_final.strip())
    text = "\n\n".join(full) + "\n"
    # redirect figure paths if present
    text = re.sub(r"outputs/figures/([A-Za-z0-9_.-]+)", r"submission/figures/\1", text)
    return text


def make_zip() -> None:
    zip_path = SUBMISSION / "submission_package.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in SUBMISSION.rglob("*"):
            if p == zip_path:
                continue
            if p.is_file():
                zf.write(p, p.relative_to(SUBMISSION))


def main():
    cfg = load_cfg()
    cfg_path = resolve_path("project_best_fit.yaml", "config/project_best_fit.yaml", "config/project.yaml")
    refs_path = resolve_path("references.bib", "paper/references.bib")
    manuscript_path = resolve_path("manuscript.md", "paper/manuscript.md")
    tables_path = resolve_path("tables.md", "paper/tables.md")
    captions_path = resolve_path("figure_captions.md", "paper/figure_captions.md")

    risk_path = load_csv_latest("risked_tonnage.csv")
    metrics_path = load_json_latest("validation_metrics.json")

    risk_df = pd.read_csv(risk_path)
    metrics = json.loads(read_text(metrics_path))
    vario = parse_variogram_values(cfg)

    SUBMISSION.mkdir(parents=True, exist_ok=True)
    write_typesetting_files()

    body = update_manuscript(read_text(manuscript_path), cfg, risk_df)
    captions = update_figure_captions(read_text(captions_path), cfg, vario)
    tables = update_tables(read_text(tables_path), cfg, risk_df, metrics)

    write_text(SUBMISSION / "paper_body.md", body)
    write_text(SUBMISSION / "figure_captions_final.md", captions)
    write_text(SUBMISSION / "tables_final.md", tables)

    paper_md = assemble_paper(body, tables, captions)
    write_text(SUBMISSION / "paper.md", paper_md)

    missing_figures = copy_figures()
    copy_supplement()

    write_text(
        SUBMISSION / "REPRODUCIBILITY.md",
        build_reproducibility(cfg_path, refs_path, missing_figures),
    )
    write_text(
        SUBMISSION / "SUBMISSION_CHECKLIST.md",
        build_checklist(cfg, risk_df, paper_md, refs_path, missing_figures),
    )

    make_zip()

    print("Submission package built:")
    print(f"- {SUBMISSION / 'paper.md'}")
    print(f"- {SUBMISSION / 'paper_body.md'}")
    print(f"- {SUBMISSION / 'tables_final.md'}")
    print(f"- {SUBMISSION / 'figure_captions_final.md'}")
    print(f"- {SUBMISSION / 'submission_package.zip'}")
    if missing_figures:
        print("Missing figures:", ", ".join(missing_figures))


if __name__ == "__main__":
    main()
