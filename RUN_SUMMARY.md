# Run Summary

## Current repo state

- The active journal package is `submission_ready/`.
- The package is clean JAES-facing output, not an internal audit archive.
- Root-level duplicate manuscript exports and control files are excluded.
- `scripts/build_submission_package.py` mirrors the clean package into this repo after a successful parent-project build.

## Main commands

Run these from the parent project folder:

```bash
python scripts/autoresearch_eval.py
python scripts/submission_preflight.py --sub-dir submission_ready
python scripts/submission_preflight.py --sub-dir repo/submission_ready
```

## Package outputs

- `01_Title_Page.docx`
- `02_Highlights.docx`
- `03_Graphical_Abstract.tif`
- `04_Manuscript.docx`
- `05_Tables.docx`
- `06_Figure_Captions.docx`
- `07_Cover_Letter.docx`
- `08_Declaration_of_Interest.docx`
- `09_Author_Statement.docx`
- numbered TIFF figures
- compact supplementary ZIP and CSV files

## Data policy

Keep private drillhole data out of tracked files. The public sample tables in `demo_data/` are demonstration inputs only.
