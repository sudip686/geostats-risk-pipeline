from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = ROOT / "output" / "a3_categorical_25_50_nr100"
DEFAULT_OUT_DIR = ROOT / "build" / "notebooklm_canonical_review_pack"
NOTEBOOK_TITLE = "N6_JAES_Reviewer_Fixes_2m_Grid"

SOURCE_ROWS = [
    {
        "slug": "G01_case_2023_graphite_creek",
        "title": "Insights into the metamorphic history and origin of flake graphite mineralization at the Graphite Creek graphite deposit, Seward Peninsula, Alaska, USA",
        "authors": "Case et al.",
        "year": "2023",
        "journal": "Mineralium Deposita",
        "doi": "10.1007/s00126-023-01161-3",
        "status": "verified_web_primary",
        "benchmark": "published_language_model",
        "role": "graphite_deposit_style",
        "allowed": [
            "Use as a style model for geology-first graphite papers that combine field observations, metamorphic context, geochemistry/petrochronology/Raman evidence, and origin interpretation.",
            "Use to support cautious language that high-grade flake graphite enrichment can be interpreted through metamorphic processes, structural focusing, and lithological contacts where evidence exists.",
        ],
        "blocked": [
            "Do not imply Tanga has equivalent Raman, isotope, petrochronology, or petrographic evidence unless those datasets are actually supplied.",
            "Do not transfer Alaska-specific ages, tectonic history, or resource scale to Tanzania.",
        ],
        "targets": ["1", "2", "5.1", "style_bank"],
    },
    {
        "slug": "G02_drever_2023_bissett_creek",
        "title": "Petrogenesis of extra-large flake graphite at the Bissett Creek deposit, Canada",
        "authors": "Drever; Kinney; Yakymchuk",
        "year": "2023",
        "journal": "Mineralium Deposita",
        "doi": "10.1007/s00126-022-01145-9",
        "status": "verified_web_primary",
        "benchmark": "published_language_model",
        "role": "graphite_deposit_style",
        "allowed": [
            "Use as a style model for reporting flake graphite hosted by upper-amphibolite-facies gneisses with emphasis on petrogenesis, host lithology, and exploration implications.",
            "Use to reinforce the need to describe graphite form, host rock, metamorphic grade, and process uncertainty before making continuity claims.",
        ],
        "blocked": [
            "Do not claim extra-large flake graphite at Tanga without flake-size evidence.",
            "Do not cite as direct African/Mozambique Belt evidence.",
        ],
        "targets": ["2", "3.3", "5.1", "style_bank"],
    },
    {
        "slug": "G03_case_2025_time_space_graphite",
        "title": "A time-space model of graphite mineral systems",
        "authors": "Case",
        "year": "2025",
        "journal": "Mineralium Deposita",
        "doi": "10.1007/s00126-025-01412-5",
        "status": "verified_web_primary",
        "benchmark": "mineral_system_anchor",
        "role": "graphite_mineral_systems",
        "allowed": [
            "Use to frame Tanga as an orogenic/metamorphic flake-graphite system rather than an isolated exploration target.",
            "Use to support language that graphitization, strain focus, anatexis, and grain-boundary-scale fluids may enhance graphite grade or flake size, but only as regional/process context.",
        ],
        "blocked": [
            "Do not turn mineral-system context into proof of local graphite genesis.",
            "Do not overclaim flake size, carbon source, or product quality without petrography/Raman/isotopes.",
        ],
        "targets": ["1", "2", "3.3", "5.1"],
    },
    {
        "slug": "G04_ancuabe_style_graphite_mozambique",
        "title": "Northern Mozambique / Ancuabe-style graphite form and fabric analogue",
        "authors": "Analogue literature note",
        "year": "various",
        "journal": "African graphite analogue source note",
        "doi": "verify_exact_source_before_citation",
        "status": "workbook_search_target",
        "benchmark": "analogue_target",
        "role": "graphite_form_fabric",
        "allowed": [
            "Use as a NotebookLM search target for graphite form and fabric language: disseminated flake graphite, foliation-parallel graphite, semi-massive graphitic schist, veinlet graphite, and weathered graphitic schist.",
            "Use only after NotebookLM has attached and cited a specific source.",
        ],
        "blocked": [
            "Do not cite this placeholder row in the manuscript.",
            "Do not import analogue observations into Tanga unless Tanga core/petrography/log evidence supports them.",
        ],
        "targets": ["3.3", "P22", "style_bank"],
    },
    {
        "slug": "B01_bai_2022_gsf_sgs",
        "title": "Sequential Gaussian simulation for geosystems modeling: A machine learning approach",
        "authors": "Bai; Tahmasebi",
        "year": "2022",
        "journal": "Geoscience Frontiers",
        "doi": "10.1016/j.gsf.2021.101258",
        "status": "verified_local_and_web",
        "benchmark": "benchmark_tier",
        "role": "method",
        "allowed": [
            "SGS is suitable for uncertainty quantification in spatial systems.",
            "Search design and computational scaling matter in large 3D SGS workflows.",
        ],
        "blocked": [
            "Do not use for Tanzania-specific geology.",
            "Do not cite as proof that the current parameters are optimal.",
        ],
        "targets": ["3.4", "4.2", "4.4"],
    },
    {
        "slug": "B02_dong_2018_gsf_usagaran",
        "title": "Petrogenesis and geochronology of granitoids from the Usagaran Belt, central Tanzania",
        "authors": "Dong et al.",
        "year": "2018",
        "journal": "Geoscience Frontiers",
        "doi": "10.1016/j.gsf.2018.03.003",
        "status": "verified_local",
        "benchmark": "benchmark_tier",
        "role": "regional_context",
        "allowed": [
            "Southern Tanzania belongs to a polyphase East African/Mozambique Belt framework.",
            "Regional context should be written in terms of polyphase deformation and metamorphism.",
        ],
        "blocked": [
            "Do not convert regional geology into deposit-scale anisotropy numbers.",
        ],
        "targets": ["2", "5.1"],
    },
    {
        "slug": "A01_maleki_emery_2020_contact_analysis",
        "title": "Geostatistics in the presence of geological boundaries: Exploratory tools for contact analysis",
        "authors": "Maleki; Emery",
        "year": "2020",
        "journal": "Ore Geology Reviews",
        "doi": "10.1016/j.oregeorev.2020.103397",
        "status": "verified_local",
        "benchmark": "anchor_only",
        "role": "contact_analysis",
        "allowed": [
            "Contact analysis is the correct diagnostic for testing combined-domain stationarity across a boundary.",
            "Fresh-weathered grouping should be tested instead of assumed.",
        ],
        "blocked": [
            "Do not call the boundary hard or soft until the plot is generated.",
        ],
        "targets": ["4.1", "4.4", "rebuttal"],
    },
    {
        "slug": "A02_emery_maleki_2019_boundaries",
        "title": "Geostatistics in the presence of geological boundaries: Application to mineral resources modeling",
        "authors": "Emery; Maleki",
        "year": "2019",
        "journal": "Ore Geology Reviews",
        "doi": "10.1016/j.oregeorev.2019.103124",
        "status": "verified_local_note",
        "benchmark": "anchor_only",
        "role": "domain_boundaries",
        "allowed": [
            "Boundary handling can materially affect resource models.",
            "Geological boundaries should be treated explicitly.",
        ],
        "blocked": [
            "Do not use this alone to defend the final domain split or merge.",
        ],
        "targets": ["3.2", "4.1", "5.3"],
    },
    {
        "slug": "A03_boisvert_deutsch_2011_lva",
        "title": "Programs for kriging and sequential Gaussian simulation with locally varying anisotropy using non-Euclidean distances",
        "authors": "Boisvert; Deutsch",
        "year": "2011",
        "journal": "Computers & Geosciences",
        "doi": "10.1016/j.cageo.2010.03.021",
        "status": "verified_local_note",
        "benchmark": "anchor_only",
        "role": "anisotropy",
        "allowed": [
            "Locally varying anisotropy is a valid future refinement.",
            "Stationary anisotropy is an approximation.",
        ],
        "blocked": [
            "Do not claim LVA is implemented if the workflow still uses a single orebody axis set.",
        ],
        "targets": ["4.2", "5.1"],
    },
    {
        "slug": "A04_deutsch_2023_minerals",
        "title": "The Place of Geostatistical Simulation through the Life Cycle of a Mineral Deposit",
        "authors": "Deutsch",
        "year": "2023",
        "journal": "Minerals",
        "doi": "10.3390/min13111400",
        "status": "verified_local",
        "benchmark": "anchor_only",
        "role": "simulation_practice",
        "allowed": [
            "Simulation is used to express uncertainty envelopes rather than single deterministic truths.",
            "Simulation belongs in resource-model decision support.",
        ],
        "blocked": [
            "Do not use for East African geology.",
        ],
        "targets": ["1", "3.4", "5.2", "6"],
    },
    {
        "slug": "A05_lindi_2024_nrr",
        "title": "Uncertainty Quantification in Mineral Resource Estimation",
        "authors": "Lindi et al.",
        "year": "2024",
        "journal": "Natural Resources Research",
        "doi": "10.1007/s11053-024-10394-6",
        "status": "verified_local_and_web",
        "benchmark": "anchor_only",
        "role": "uncertainty_review",
        "allowed": [
            "Uncertainty accumulates stepwise through the resource estimation workflow.",
            "Method choice should fit deposit characteristics and data support.",
        ],
        "blocked": [
            "Do not use to justify exact search radii or anisotropy ratios.",
        ],
        "targets": ["1", "3.1", "5.2"],
    },
    {
        "slug": "A06_case_2025_graphite_systems",
        "title": "A time-space model of graphite mineral systems",
        "authors": "Case",
        "year": "2025",
        "journal": "Mineralium Deposita",
        "doi": "10.1007/s00126-025-01412-5",
        "status": "verified_local_and_web",
        "benchmark": "anchor_only",
        "role": "graphite_geology",
        "allowed": [
            "Graphite systems can be framed in an orogenic metasedimentary context.",
            "Metamorphic and structural concentration can both matter in graphite systems.",
        ],
        "blocked": [
            "Do not use it as direct evidence for the local weathering-grade jump magnitude.",
        ],
        "targets": ["2", "3.3", "4.1", "4.2"],
    },
    {
        "slug": "A07_fritz_2013_eao_review",
        "title": "Orogen styles in the East African Orogen: A review of the Neoproterozoic to Cambrian tectonic evolution",
        "authors": "Fritz et al.",
        "year": "2013",
        "journal": "Journal of African Earth Sciences",
        "doi": "10.1016/j.jafrearsci.2013.06.004",
        "status": "verified_local_and_web",
        "benchmark": "anchor_only",
        "role": "regional_context",
        "allowed": [
            "The East African Orogen is polyphase and tectonically complex.",
            "Southern Tanzania belongs to the broader belt framework.",
        ],
        "blocked": [
            "Do not treat belt-scale review statements as local range measurements.",
        ],
        "targets": ["2", "5.1"],
    },
    {
        "slug": "A08_sommer_kroner_2013_lithos",
        "title": "Ultra-high temperature granulite-facies metamorphic rocks from the Mozambique belt of SW Tanzania",
        "authors": "Sommer; Kröner",
        "year": "2013",
        "journal": "Lithos",
        "doi": "10.1016/j.lithos.2013.02.014",
        "status": "verified_local_and_web",
        "benchmark": "anchor_only",
        "role": "metamorphic_context",
        "allowed": [
            "High-grade metamorphism is regionally relevant in SW Tanzania.",
        ],
        "blocked": [
            "Do not cite it as direct proof of Ruangwa-specific D2 shortening ratios.",
        ],
        "targets": ["2", "4.2"],
    },
    {
        "slug": "A09_thomas_2014_ruangwa",
        "title": "Geochronology of granitic rocks from the Ruangwa region, southern Tanzania: links with NE Mozambique and beyond",
        "authors": "Thomas et al.",
        "year": "2014",
        "journal": "Journal of African Earth Sciences",
        "doi": "10.1016/j.jafrearsci.2014.06.012",
        "status": "verified_local",
        "benchmark": "anchor_only",
        "role": "ruangwa_anchor",
        "allowed": [
            "Ruangwa is the correct named regional anchor in southern Tanzania.",
        ],
        "blocked": [
            "Do not treat it as a graphite deposit paper.",
        ],
        "targets": ["2"],
    },
    {
        "slug": "A10_barnett_2026_probabilistic_disclosure",
        "title": "Mineral Resource Disclosure with Probabilistic Models",
        "authors": "Barnett et al.",
        "year": "2026",
        "journal": "Mathematical Geosciences",
        "doi": "10.1007/s11004-025-10244-y",
        "status": "verified_local",
        "benchmark": "anchor_only",
        "role": "probabilistic_reporting",
        "allowed": [
            "P10/P50/P90 style reporting fits probabilistic resource disclosure logic.",
        ],
        "blocked": [
            "Do not cite this as a Gondwana Research structural geology paper.",
        ],
        "targets": ["4.3", "5.2", "6"],
    },
    {
        "slug": "A11_chiles_delfiner_2012_geostatistics",
        "title": "Geostatistics: Modeling Spatial Uncertainty, 2nd edition",
        "authors": "Chiles; Delfiner",
        "year": "2012",
        "journal": "Wiley book",
        "doi": "book_reference",
        "status": "standard_method_source",
        "benchmark": "method_anchor",
        "role": "geostatistical_foundation",
        "allowed": [
            "Use for general spatial uncertainty, variogram, and change-of-support wording.",
            "Use to keep SGS described as an uncertainty-transfer method rather than proof of geological truth.",
        ],
        "blocked": [
            "Do not use as project-specific validation evidence.",
        ],
        "targets": ["3.4", "4.4", "5.2"],
    },
    {
        "slug": "A12_goovaerts_1997_geostatistics",
        "title": "Geostatistics for Natural Resources Evaluation",
        "authors": "Goovaerts",
        "year": "1997",
        "journal": "Oxford University Press book",
        "doi": "book_reference",
        "status": "standard_method_source",
        "benchmark": "method_anchor",
        "role": "geostatistical_foundation",
        "allowed": [
            "Use for standard conditional simulation and validation language.",
            "Use to describe realizations as equiprobable outcomes under assumptions.",
        ],
        "blocked": [
            "Do not use as evidence that the Tanga model validates well.",
        ],
        "targets": ["3.4", "4.4", "5.2"],
    },
]

UNRESOLVED = [
    ["Emmiru et al. (2025), Earth-Science Reviews", "unresolved", "Not found in the verified local corpus for this pass. Do not cite until a title/DOI/PDF is supplied."],
    ["Sun et al. (2024), Earth-Science Reviews", "unresolved", "Keep only as a search target. Not approved as evidence."],
    ["Lemos et al. (2023), Nature Communications / Communications Earth & Environment", "unresolved", "Exact paper not confidently resolved for the requested claim."],
    ["Francis (2021), International Journal of Mining Science and Technology", "unresolved", "Exact nugget-decomposition paper not verified locally."],
]

PROMPTS = [
    ("P01", "contact_analysis", "Use `tables/contact_analysis.csv` and `tables/contact_analysis_meta.json` to write a 180-220 word fresh-vs-weathered contact-analysis paragraph. Label each sentence as [Direct], [Proxy], or [Hypothesis]."),
    ("P02", "domaining_gate", "Draft a short decision note on whether GRSC and SAPR stay combined or should be split. Format: Observation, Interpretation, Modeling implication, Risk if wrong."),
    ("P03", "z_swath_support_defense", "Use `tables/support_ladder_summary.csv`, `tables/vertical_continuity_summary.json`, `validation_metrics.json`, and `validation_metrics_2m.json` to write a technical defense of the vertical swath. If the 2 m Z-swath improves materially, describe it as evidence of a support effect and avoid the word 'prove'."),
    ("P04", "support_figure_caption", "Draft a caption for a two-panel Z-swath figure comparing reporting support and simulation support."),
    ("P05", "nugget_decomposition", "Use `nugget_decomposition.csv` to write a short paragraph comparing raw-grade and log-grade nugget behavior. Interpret lower log-nugget behavior as evidence of proportional effect or clustering only if supported by the table."),
    ("P06", "nsr_selectivity_note", "Write a short internal note on what the nugget-to-sill ratio implies for confidence in selective local partitioning at current drill spacing."),
    ("P07", "anisotropy_interpretation", "Write a 150-180 word anisotropy paragraph using the range ordering plus a structural-complexity caveat. Mention locally varying anisotropy only as future work."),
    ("P08", "compositing_support", "Draft a 130-170 word paragraph explaining why 2.0 m composites are a cautious local data-support choice. Keep any lamination wording as [Proxy] unless measured thickness data are supplied."),
    ("P09", "weathering_upgrade", "Use `weathering_summary.csv` and contact-analysis outputs to write a paragraph on weathering-related grade concentration. Do not say weathering created graphite."),
    ("P10", "confidence_gradient", "Use the confidence-gradient tables to explain where P10-P90 spread is widest and what that means for drilling priorities."),
    ("P10B", "domain_uncertainty", "Use `tables/domain_uncertainty_summary.json`, `tables/domain_uncertainty_hotspots.csv`, and `figures/domain_entropy_map.png` to explain where boundary position is stable versus uncertain. Keep this as boundary uncertainty, not as proof of structure."),
    ("P11", "risk_aperture", "Write a paragraph on the widening risk aperture at higher cutoffs. Use the phrase 'scale-invariant heterogeneity' rather than 'fractal'."),
    ("P11B", "thickness_geometry", "Use `tables/thickness_geometry_summary.json` and `tables/thickness_geometry_hotspots.csv` to explain how graphitic thickness/geometry uncertainty differs from grade uncertainty under the current structural frame."),
    ("P12", "boudinage_hypothesis", "Draft a short interpretive paragraph on whether the strike-to-dip range contrast is consistent with lens continuity or boudinage-like segmentation. Mark boudinage as [Hypothesis]."),
    ("P13", "methods_refresh", "Rewrite the methods summary so it states 2.0 m compositing, 25 x 25 x 2 m simulation support, 50 x 50 x 2 m reporting support, and a 360 x 160 x 70 m fixed search ellipsoid."),
    ("P14", "validation_rewrite", "Rewrite Section 4.4 as: distribution diagnostics, lateral swaths, vertical swath, implication for model use. Use exact metrics from the current output files."),
    ("P15", "discussion_oii", "Rewrite the discussion in the pattern Observation -> Interpretation -> Area-of-interest implication. Put direct observations first."),
    ("P16", "conclusions_refresh", "Write a 130-160 word conclusions paragraph using the phrases 'geology-led' and 'fabric-concordant anisotropy'. End with where vertical refinement is still required."),
    ("P17", "reviewer_rebuttal_vertical", "Draft a reviewer-response paragraph for the low Z-swath concern using only current metrics and claim-gated wording."),
    ("P18", "reviewer_rebuttal_domains", "Draft a reviewer-response paragraph for the fresh/weathered domain-mixing concern using contact-analysis outputs."),
    ("P19", "claim_evidence_map", "Generate a claim-evidence map with columns: claim, evidence class, source/output, approved wording. Reject unsupported claims."),
    ("P20", "reference_hygiene", "Audit the manuscript references against `reference_resolution.csv`. List approved, replace, and quarantined citations."),
    ("P21", "final_synthesis", "Using all verified source notes plus the run-output snapshot, write a final synthesis note titled 'JAES 2m reviewer-fix narrative'."),
    ("P22", "published_style_rewrite", "Using only the attached published-paper style notes, rewrite the Introduction, geology-to-prior section, and Discussion opening in the style of reputable geology papers: observation first, interpretation second, uncertainty third, transferable significance last. Do not imitate sentences verbatim. Avoid promotional resource language."),
    ("P23", "graphite_form_gap", "List every graphite-form claim that requires core photography, petrography, Raman, XRD, isotope, or flake-size evidence. For unsupported claims, provide safer replacement wording."),
    ("P24", "geology_conditioned_sgs_claim", "Draft a 120-160 word paragraph explaining the no-domain isotropic pilot versus geology-conditioned SGS comparison. Treat the pilot as sensitivity evidence only and state that better global metrics alone do not prove better geological realism."),
    ("P25", "jaes_scope_language", "Draft three JAES-safe topic sentences that connect the local Tanga case to broader African Earth-science significance without using largest/largest-deposit/resource-promotion language."),
]


STYLE_BANK = """# Published-Paper Wording Bank for JAES Revision

Use this as a source inside NotebookLM to guide tone and sentence architecture. It is not a citation source by itself; it is a writing-control note distilled from reputable geology and geostatistics papers.

## Required Pattern

1. Begin with the geological object, not the software.
2. State what is observed in the project data.
3. State the interpretation as a testable prior.
4. State how SGS tests or transfers that prior into uncertainty space.
5. State the limitation before the conclusion.

## Approved Sentence Frames

- `The Tanga case is treated here as a deposit-scale expression of a high-grade graphitic metasedimentary system, not as an isolated interpolation exercise.`
- `The modelling choices are therefore geological hypotheses: fabric-parallel continuity, contact-localized uncertainty, and weak thickness-normal persistence.`
- `Conditional simulation is used to test how these hypotheses behave between drill sections; it is not treated as an independent source of geological truth.`
- `The comparison with a no-domain isotropic pilot is informative but not decisive, because global histogram and Q-Q metrics can improve while geological realism declines.`
- `The main geological result is hierarchical: lateral continuity is more stable than thickness-normal continuity, and uncertainty is concentrated where contacts, weathering state, and package geometry are least resolved.`
- `In the absence of Raman, isotope, XRD, and systematic petrographic evidence, graphite genesis and product-quality claims are kept interpretive and provisional.`

## Words to Prefer

- `deposit-scale expression`
- `graphitic-schist architecture`
- `fabric-concordant prior`
- `conditional test`
- `uncertainty transfer`
- `screening-level diagnostic`
- `contact-localized uncertainty`
- `thickness-normal risk`
- `regional mineral-system context`

## Words to Avoid

- `confirmed`
- `largest`
- `resource tonnage`
- `proven continuity`
- `validated model` unless metrics are strong
- `economic potential` in the main scientific conclusion
- `the SGS shows the geology` without specifying the conditioning assumptions

## Model Paragraph

The Tanga graphite system is framed as a graphitic-schist-hosted, stratiform metamorphic graphite occurrence within the Tanzanian Mozambique Belt. The key modelling question is not whether a stochastic workflow can produce a smooth grade envelope, but whether the mapped and logged geological architecture provides defensible priors for continuity. Three priors are therefore tested: continuity should be strongest within the foliation-parallel graphitic package, uncertainty should increase near contacts and weathering transitions, and thickness-normal continuity should be the least stable component of the model. Sequential Gaussian simulation is used as a conditional uncertainty-transfer tool to evaluate these priors at reporting support. The resulting model is interpreted cautiously because distributional and swath validation remain moderate to weak; the emphasis is therefore on the spatial organization of uncertainty, not on precise grade prediction or resource-style reporting.
"""


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def read_csv(path: Path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def note_text(row: dict[str, object]) -> str:
    allowed = "\n".join(f"- {x}" for x in row["allowed"])
    blocked = "\n".join(f"- {x}" for x in row["blocked"])
    targets = "\n".join(f"- {x}" for x in row["targets"])
    return (
        f"# {row['title']}\n\n"
        f"- Authors: {row['authors']}\n"
        f"- Year: {row['year']}\n"
        f"- Journal: {row['journal']}\n"
        f"- DOI: {row['doi']}\n"
        f"- Status: {row['status']}\n"
        f"- Benchmark: {row['benchmark']}\n"
        f"- Role: {row['role']}\n\n"
        "## Allowed Claims\n\n"
        f"{allowed}\n\n"
        "## Forbidden Overclaims\n\n"
        f"{blocked}\n\n"
        "## Section Targets\n\n"
        f"{targets}\n"
    )


def run_snapshot_text(run_dir: Path) -> str:
    tables = run_dir / "tables"
    reporting = read_json(tables / "validation_metrics.json")
    sim = read_json(tables / "validation_metrics_2m.json")
    support = read_csv(tables / "support_ladder_summary.csv")
    vertical = read_json(tables / "vertical_continuity_summary.json")
    weather = read_csv(tables / "weathering_summary.csv")
    nugget = read_csv(tables / "nugget_decomposition.csv")
    conf = read_json(tables / "confidence_gradient_meta.json")
    domain_unc = read_json(tables / "domain_uncertainty_summary.json")
    thickness = read_json(tables / "thickness_geometry_summary.json")
    lines = [f"# Run Output Snapshot", "", f"- Run directory: `{run_dir}`", ""]
    if reporting:
        lines += [
            "## Reporting Support",
            "",
            f"- Histogram overlap: `{reporting.get('hist_overlap')}`",
            f"- Q-Q RMSE: `{reporting.get('qq_rmse')}`",
            f"- Swath X/Y/Z: `{reporting.get('swath_corr_x')}` / `{reporting.get('swath_corr_y')}` / `{reporting.get('swath_corr_z')}`",
            "",
        ]
    else:
        lines += ["## Reporting Support", "", "- Pending: `validation_metrics.json` not found.", ""]
    if sim:
        lines += [
            "## Simulation Support",
            "",
            f"- Histogram overlap: `{sim.get('hist_overlap')}`",
            f"- Q-Q RMSE: `{sim.get('qq_rmse')}`",
            f"- Swath X/Y/Z: `{sim.get('swath_corr_x')}` / `{sim.get('swath_corr_y')}` / `{sim.get('swath_corr_z')}`",
            "",
        ]
    else:
        lines += ["## Simulation Support", "", "- Pending: `validation_metrics_2m.json` not found.", ""]
    if support:
        lines += ["## Support Ladder", ""] + [
            f"- {r['support_name']}: swath X/Y/Z = `{r['swath_corr_x']}` / `{r['swath_corr_y']}` / `{r['swath_corr_z']}` at `{r['support_dx_m']} x {r['support_dy_m']} x {r['support_dz_m']}` m"
            for r in support
        ] + [""]
    if vertical:
        lines += [
            "## Vertical Continuity Summary",
            "",
            f"- Plane mean swath corr: `{vertical.get('plane_mean_swath_corr')}`",
            f"- Normal-direction swath corr: `{vertical.get('normal_direction_swath_corr')}`",
            f"- Normal-to-plane ratio: `{vertical.get('normal_to_plane_ratio')}`",
            "",
        ]
    if weather:
        lines += ["## Weathering", ""] + [f"- {r['group']}: mean `{r['mean_tgc_pct']}`, count `{r['count']}`" for r in weather] + [""]
    if nugget:
        lines += ["## Nugget", ""] + [f"- {r['transform']}: nugget ratio `{r['nugget_ratio']}`" for r in nugget] + [""]
    if domain_unc:
        lines += [
            "## Domain Uncertainty",
            "",
            f"- Mean entropy: `{domain_unc.get('mean_domain_entropy')}`",
            f"- P90 entropy: `{domain_unc.get('p90_domain_entropy')}`",
            f"- Cells max probability < 0.70 (%): `{domain_unc.get('cells_max_probability_lt_0_70_pct')}`",
            "",
        ]
    if thickness:
        lines += [
            "## Thickness / Geometry",
            "",
            f"- Mean P50 graphitic thickness (m): `{thickness.get('mean_p50_graphitic_thickness_m')}`",
            f"- Cells aperture >= 100%: `{thickness.get('cells_aperture_ge_100pct')}`",
            "",
        ]
    if conf:
        lines += [
            "## Confidence Gradient",
            "",
            f"- Max risk aperture pct: `{conf.get('max_risk_aperture_pct')}`",
            f"- Median risk aperture pct: `{conf.get('median_risk_aperture_pct')}`",
            "",
        ]
    return "\n".join(lines)


def build(run_dir: Path, out_dir: Path) -> dict[str, object]:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    sources = out_dir / "sources"
    gists = sources / "gists"
    prompts = out_dir / "prompts"

    write(out_dir / "README.md", f"# N6 JAES 2m Workbook\n\nRecommended Notebook title: `{NOTEBOOK_TITLE}`")
    preface_sources = [
        (
            sources / "00_run_output_snapshot.md",
            run_snapshot_text(run_dir),
        ),
        (
            sources / "01_claim_gate_protocol.md",
            "# Claim-Gating Protocol\n\n- Direct: project output.\n- Proxy: project output interpreted with a verified source.\n- Hypothesis: plausible but not demonstrated.\n\nReject wording that turns Proxy or Hypothesis into fact.",
        ),
        (
            sources / "02_claim_gate_examples.md",
            "# Claim-Gate Examples\n\n- Direct: the 2 m support rerun changed the Z-swath correlation.\n- Proxy: the support contrast is consistent with change-of-support smoothing.\n- Hypothesis: high-cutoff hotspots may reflect localized structural thickening.",
        ),
        (
            sources / "03_structural_complexity_caveat.md",
            "# Structural Complexity Caveat\n\nThe current workflow uses a stationary orebody-axis ellipsoid. This is a useful first-order simplification, but tight hinge zones may require locally varying anisotropy in future iterations.",
        ),
        (
            sources / "04_nsr_selectivity_note.md",
            "# Nugget-to-Sill Ratio Decision Note\n\nUse the NSR to describe short-scale uncertainty limits, not to force categorical mine-method claims.",
        ),
        (
            sources / "05_support_comparison_note.md",
            "# Support Comparison Note\n\nKeep these separate:\n- composite support: 2.0 m\n- simulation support: 25 x 25 x 2 m\n- reporting support: 50 x 50 x 2 m",
        ),
        (
            sources / "06_surgical_insert_candidates.md",
            "# Surgical Insert Candidates\n\n- Section 3.3: composite-support rationale\n- Section 4.1: contact-analysis result\n- Section 4.2: search ellipsoid and structural caveat\n- Section 4.4: reporting-support vs simulation-support validation\n- Section 5.1: domain entropy and boundary stability\n- Section 5.2: thickness / geometry risk under the current structural frame\n- Section 5.3: confidence gradient and drill-targeting note",
        ),
        (
            sources / "08_published_paper_wording_bank.md",
            STYLE_BANK,
        ),
    ]

    manifest_rows = []
    for path, text in preface_sources:
        write(path, text)
        manifest_rows.append({"path": str(path), "title": path.stem, "kind": "source"})

    for row in SOURCE_ROWS:
        path = gists / f"{row['slug']}.md"
        write(path, note_text(row))
        manifest_rows.append({"path": str(path), "title": path.stem, "kind": "source"})

    unresolved_lines = ["# Requested But Unverified References", ""]
    for label, status, note in UNRESOLVED:
        unresolved_lines += [f"## {label}", "", f"- Status: {status}", f"- Note: {note}", ""]
    unresolved_path = sources / "07_unverified_requested_sources.md"
    write(unresolved_path, "\n".join(unresolved_lines))
    manifest_rows.append({"path": str(unresolved_path), "title": unresolved_path.stem, "kind": "source"})

    for pid, slug, prompt in PROMPTS:
        write(prompts / f"{pid}_{slug}.md", f"# {pid}\n\n{prompt}")

    ref_rows = []
    for row in SOURCE_ROWS:
        ref_rows.append(
            {
                "requested_label": f"{row['authors']} ({row['year']})",
                "resolved_title": row["title"],
                "journal": row["journal"],
                "doi": row["doi"],
                "status": row["status"],
                "benchmark": row["benchmark"],
                "role": row["role"],
                "targets": "; ".join(row["targets"]),
                "workbook_use": "approved",
            }
        )
    for label, status, note in UNRESOLVED:
        ref_rows.append(
            {
                "requested_label": label,
                "resolved_title": "",
                "journal": "",
                "doi": "",
                "status": status,
                "benchmark": "unresolved",
                "role": "blocked",
                "targets": "",
                "workbook_use": note,
            }
        )

    with (out_dir / "reference_resolution.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ref_rows[0].keys()))
        writer.writeheader()
        writer.writerows(ref_rows)

    with (out_dir / "notebooklm_sources_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "title", "kind"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    ps = [f"$NotebookTitle = '{NOTEBOOK_TITLE}'", "$Notebook = notebooklm create $NotebookTitle --json | ConvertFrom-Json", "$NotebookId = $Notebook.id", ""]
    for row in manifest_rows:
        ps.append(f"notebooklm source add '{row['path']}' -n $NotebookId --title '{row['title']}' --json")
    write(out_dir / "notebooklm_add_commands.ps1", "\n".join(ps))

    status = {"notebook_title": NOTEBOOK_TITLE, "run_dir": str(run_dir), "source_count": len(manifest_rows), "prompt_count": len(PROMPTS)}
    write(out_dir / "manifest.json", json.dumps(status, indent=2))
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the JAES 2 m NotebookLM workbook.")
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    print(json.dumps(build(run_dir, out_dir), indent=2))


if __name__ == "__main__":
    main()
