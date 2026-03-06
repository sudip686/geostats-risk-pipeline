# geostats-risk-pipeline (repo package)

This `repo/` folder is aligned with the local workflow code and is runnable as a standalone package.

## Scope
- Full workflow modules in `src/`:
  - validation, desurvey, compositing, domain filtering, declustering, normal-score transform
  - variography, SGS, risk postprocessing, validation plots
  - drill-spacing sensitivity and internal validation hooks
- Build/analysis utilities in `scripts/`:
  - paper build/export, zip packaging, validation and diagnostics helpers
- Manuscript assets in `paper/` and root paper files (`manuscript.md`, `tables.md`, `figure_captions.md`)
- Portable demo dataset in `demo_data/`

## Data-source parameterization
Set in `config/project.yaml` (or `config/project_best_fit.yaml`):

```yaml
data_source: repo  # repo/@repo/demo -> paths.repo_data_dir, local -> paths.local_data_dir
paths:
  local_data_dir: data
  repo_data_dir: demo_data
```

Resolution rules:
- `repo`, `@repo`, or `demo` -> `paths.repo_data_dir`
- `local` -> `paths.local_data_dir`

## Repo-mode guardrails
For portability, when `data_source` is `repo/demo` and external files are not present:
- calibration is auto-disabled if `calibration.reference_data` is missing
- external validation reference path is removed if missing
- internal block-model validation is auto-disabled if `internal_validation.model_csv` is missing

These safeguards do not change local-mode behavior when local files are provided.

## Run workflow
From inside `repo/`:

```bash
python -m src.run_all --config config/project.yaml --output outputs
```

Compatibility entrypoint:

```bash
python -m src.run --config config/project.yaml --output outputs
```

Use local project data:
1. Put input CSVs in `data/`
2. Set `data_source: local`
3. Re-run the same command

## Build paper package
From inside `repo/`:

```bash
python scripts/build_paper.py
bash scripts/export_docx.sh
bash scripts/export_pdf.sh
```

## Push hygiene
- Keep confidential datasets out of git.
- Keep `demo_data/` as non-confidential sample inputs.
- Keep repo default as `data_source: repo` for public reproducibility.
