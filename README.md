# geostats-risk-pipeline

Reproducible SGS/kriging workflow with synthetic drillhole demo data, configs, and pinned environment for risk/uncertainty analysis. Real drillhole data remain confidential.

## Quick start
```bash
conda env create -f environment.yml
conda activate geostats-risk-pipeline
python -m src.run --config config/project.yaml --data demo_data
```

## Repository layout
- `src/`: SGS pipeline entry point (`run.py`) — replace `stub_sgs` with your actual simulation/kriging logic.
- `config/project.yaml`: grid, variogram, RNG seeds, SGS options, paths.
- `demo_data/`: tiny synthetic drillhole tables (collar, survey, assay, litho).
- `outputs/`: results written here (kept empty in git via `.gitkeep`).
- `sgs_meta.json`: static metadata template for your paper/release.
- `RUN_SUMMARY.md`: commands, inputs, outputs for reproducibility log.
- `environment.yml`: pinned dependencies.

## Data availability statement
Raw drillhole data are proprietary/confidential. Synthetic example data, full code, configs, and environment needed to reproduce the workflow are provided.

## How to adapt for publication
1. Replace `stub_sgs` in `src/run.py` with your SGS/kriging workflow (transform, variogram, simulation, back-transform, export).
2. Update `config/project.yaml` and `sgs_meta.json` with final parameters and commit hash.
3. Add derived outputs to `outputs/` as needed; keep raw/proprietary data out of the repo.
4. Create a tagged release and archive (zip/tar.gz) for Zenodo to obtain a DOI.

## Manuscript-ready citation text
- Code/config: "Code and configuration used in this study are available at <DOI/link>."
- Data: "Raw drillhole data are proprietary; synthetic example data, derived outputs, and full workflow materials are provided for reproducibility."
