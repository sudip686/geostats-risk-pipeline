# geostats-risk-pipeline

Reproducible SGS/kriging workflow with synthetic drillhole demo data, configuration templates, and pinned environment for risk and uncertainty analysis. Raw drillhole data remain confidential; only synthetic examples are included.

## Features
- Clear, end-to-end pipeline skeleton (`src/run.py`) with stubs for declustering, normal-score transform, variogram fitting, SGS simulation, back-transform, and export.
- Publication-ready configuration and metadata templates (`config/project.yaml`, `sgs_meta.json`).
- Synthetic demo dataset (`demo_data/`) mirroring real drillhole table schemas.
- Pinned environment (`environment.yml`) for reproducibility.
- Run summary and metadata outputs in `outputs/` for reviewers.

## Requirements
- Conda (recommended) or Python 3.11 with `pip`.
- Packages listed in `environment.yml` (includes `numpy`, `pandas`, `scipy`, `gstools`, `pykrige`, `pyproj`, `matplotlib`, `tqdm`, `pyyaml`).

## Install
```bash
conda env create -f environment.yml
conda activate geostats-risk-pipeline
```
(Or create a venv and `pip install -r requirements.txt` if you generate one.)

## Run the demo
```bash
python -m src.run --config config/project.yaml --data demo_data
```
Outputs: `outputs/sgs_meta_run.json` (summary + variogram/SGS settings). Demo uses synthetic data and stubbed SGS logic.

## Use with your real data
1) Place your real CSVs under `data/` (or any folder you prefer).
2) Edit `config/project.yaml` to point to them:
```
data:
  collar: data/collar.csv
  survey: data/survey.csv
  assay: data/assay.csv
  litho: data/litho.csv
```
3) Update grid, variogram, SGS parameters in the same file.
4) Run the same command, changing `--data` if needed.

## Pipeline structure (stub contracts)
- `decluster(frames, cfg)`: return reweighted samples (add `w_declust` or similar).
- `normal_score_transform(frames, cfg)`: add normal-score column(s) and store back-transform info.
- `fit_variogram(frames, cfg)`: fit or validate variogram; return model dict/object.
- `run_sgs(frames, cfg, variogram)`: perform SGS; write or return realizations/mean/variance.
- `back_transform(results)`: convert simulations back to grade space.
- `export_results(results, outputs_dir)`: write `sgs_meta_run.json` and any grids/realizations.
Replace these stubs with your production logic; keep signatures the same.

## Config reference (key fields)
- `project.name`, `project.seed`: identifiers and RNG seed.
- `data.*`: paths to input tables.
- `grid.origin`, `grid.spacing`, `grid.dims`, `grid.rotation_deg`, `grid.ellipsoid` (search radii).
- `transform.normal_score`, `transform.decluster.cell_size`.
- `variogram.model.structures`: list of nested structures with `type`, `sill`, `range`, `anisotropy`.
- `sgs.*`: realizations, neighbors, search radii, minimum data, kriging type, mean.
- `outputs.path`: where results are written.

## Input table schemas (expected columns)
- `collar.csv`: `hole_id,x,y,z,azimuth,dip,length`
- `survey.csv`: `hole_id,from,to,azimuth,dip`
- `assay.csv`: `hole_id,from,to,grade`
- `litho.csv`: `hole_id,from,to,litho_code`

## Reproducibility & citation
- Use `RUN_SUMMARY.md` to log commands, inputs, outputs, and key parameters.
- `sgs_meta.json` holds static metadata for your release; update commit hash and final params.
- Tag a release (e.g., `v0.1.0`), archive it, and upload to Zenodo to mint a DOI.
- Manuscript text:
  - Code/config: "Code and configuration used in this study are available at <DOI/link>."
  - Data: "Raw drillhole data are proprietary; synthetic example data, derived outputs, and full workflow materials are provided for reproducibility."

## What to exclude
- Do not commit proprietary drillhole data. Keep only synthetic/demo data and derived, shareable outputs.

## Support
Open an issue on GitHub for questions or to track enhancements.
