# RUN_SUMMARY

## Current repo state
- `repo/` now carries the same workflow modules used in local runs.
- `python -m src.run_all` and `python -m src.run` both execute end-to-end from this folder.
- Input data location is parameterized by `data_source` and `paths` in config.

## Main commands
```bash
python -m src.run_all --config config/project.yaml --output outputs
python -m src.run --config config/project.yaml --output outputs
```

## Data-source behavior
- `data_source: repo`, `@repo`, or `demo` -> `paths.repo_data_dir` (default `demo_data`)
- `data_source: local` -> `paths.local_data_dir` (default `data`)

## Repo-mode portability safeguards
When running demo/repo mode without private external files:
- calibration step auto-disables if the calibration reference CSV is missing
- validation external reference path is dropped if missing
- internal block-model validation auto-disables if `internal_validation.model_csv` is missing

## Demo-data schema (`demo_data/`)
- `collar.csv`: `hole_id,x,y,z,total_depth,azimuth_deg,dip_deg`
- `survey.csv`: `hole_id,depth,azimuth_deg,dip_deg`
- `assay.csv`: `hole_id,from_m,to_m,tgc_pct`
- `lithology.csv`: `hole_id,from_m,to_m,lith_code`

## Paper/build scripts
- Build assembled paper package: `python scripts/build_paper.py`
- Export DOCX: `bash scripts/export_docx.sh`
- Export PDF (if XeLaTeX installed): `bash scripts/export_pdf.sh`

## Git push notes
- Keep private/confidential data out of tracked files.
- Keep `demo_data/` as reproducible public sample data.
- Keep repo default config on `data_source: repo`.
