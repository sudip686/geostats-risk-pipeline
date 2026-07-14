# Files Map

This mirror keeps the active source files and the clean JAES submission package. It does not keep old root-level submission exports.

| Path | Purpose | Type | Status |
|---|---|---|---|
| `README.md` | Repository overview | manual | keep |
| `GUIDE.md` | Regeneration and validation commands | manual | keep |
| `FILES_MAP.md` | Active-file map | manual | keep |
| `manuscript.md` | Maintained manuscript source mirror | generated mirror | keep |
| `tables.md` | Maintained table source mirror | generated mirror | keep |
| `figure_captions.md` | Maintained caption source mirror | generated mirror | keep |
| `submission_ready/` | Clean JAES-facing submission package | generated | keep |
| `demo_data/` | Public sample input tables only | manual/sample | keep |
| `scripts/` | Mirrored build and validation scripts | code | keep |
| `src/` | Geostatistical workflow source | code | keep |
| root-level old manuscript exports | stale generated files | generated | remove |

Regenerate files through the parent workflow instead of editing generated package artifacts by hand.
