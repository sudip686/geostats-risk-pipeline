# Tanga Graphite Geological-Uncertainty Workflow

This repository contains the source code and manuscript-generation workflow for a geology-conditioned sequential Gaussian simulation (SGS) study of a Tanzanian stratiform graphite system.

## Scope

The repository is intentionally source-only. Proprietary drillhole inputs, archived block-model tables, SGS arrays, checkpoints, build products, submission files, and journal figures are excluded. The included `demo_data/` tables are small non-project examples for exercising the pipeline structure; they do not reproduce the study results.

The associated manuscript frames geological conditioning as a way to separate graphitic support, contact uncertainty, weathering state, reporting-envelope effects, and TGC uncertainty. It does not provide a Mineral Resource estimate, a public data release, or a calibrated local prediction product.

## Contents

- `src/`: desurveying, compositing, domains, variography, SGS, validation, and post-processing modules.
- `scripts/`: run orchestration, manuscript/package building, and preflight checks.
- `config/main_config.yaml`: active project configuration; adapt it only to data you are authorised to use.
- `manuscript.md`, `tables.md`, `figure_captions.md`: generated scientific source text maintained by the package generator.
- `demo_data/`: minimal public demonstration inputs.

## Environment

Create a Python environment using the supplied `environment.yml` or `requirements.txt`:

```powershell
conda env create -f environment.yml
conda activate geostats-risk-pipeline
```

## Running a Study

Provide authorised assay, collar, lithology, and survey inputs through the configured data directory, then run:

```powershell
python -m src.run_all --config config/main_config.yaml --output output/my_run
```

The workflow uses realization checkpoints. To resume an interrupted run, repeat the command with the same output directory. Generated output is deliberately ignored by Git.

## Rebuilding Manuscript Artifacts

With an authorised completed run directory and the locally maintained submission templates, the manuscript package can be generated with:

```powershell
python scripts/build_submission_package.py --strict --run-dir output/my_run
python scripts/submission_preflight.py --sub-dir submission
```

Do not publish the resulting package or project data without the relevant permissions.

## Reproducibility Boundary

The repository enables review of the computational workflow. Full numerical regeneration of the study requires proprietary drillhole/domain inputs and authorised archived support data, which are not distributed here. Audit-level metrics and methodological details should be cited from the journal submission's Online Resources rather than inferred from the demo data.

## License and Citation

No open-data or open-code licence has been assigned in this repository yet. Obtain author approval before reusing it beyond review, replication planning, or collaboration.
