# Guide

## Regeneration

Run the parent project workflow from `C:\Users\SUDIPTA CHANDA\OneDrive\Desktop\Tanga_New`:

```bash
python scripts/autoresearch_eval.py
```

For package validation only:

```bash
python scripts/submission_preflight.py --sub-dir submission_ready
python scripts/submission_preflight.py --sub-dir repo/submission_ready
```

## Active layout

- `submission_ready/`: clean JAES package mirror.
- `manuscript.md`: maintained manuscript source mirror.
- `tables.md`: maintained table source mirror.
- `figure_captions.md`: maintained figure-caption source mirror.
- `demo_data/`: public sample inputs only.
- `scripts/`: mirrored build and validation scripts.
- `src/`: geostatistical workflow source.

Root-level manuscript exports such as old `paper` files, audit reports, and duplicate DOCX files are intentionally excluded.
