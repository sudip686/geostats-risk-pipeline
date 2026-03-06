#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUB_DIR="$ROOT_DIR/submission"
PAPER_MD="$SUB_DIR/paper.md"
NOTE_FILE="$SUB_DIR/PDF_EXPORT_NOTE.txt"

if ! command -v pandoc >/dev/null 2>&1; then
  echo "pandoc not found. Install pandoc to export PDF."
  exit 1
fi

if [ ! -f "$PAPER_MD" ]; then
  echo "Missing $PAPER_MD. Run: python scripts/build_paper.py"
  exit 1
fi

BIB_FILE="$ROOT_DIR/references.bib"
META_FILE="$SUB_DIR/paper.yaml"
TEMPLATE_FILE="$SUB_DIR/template.tex"
if [ ! -f "$BIB_FILE" ]; then
  BIB_FILE="$ROOT_DIR/paper/references.bib"
fi

if [ ! -f "$BIB_FILE" ]; then
  echo "references.bib not found at root or paper/."
  exit 1
fi

if ! command -v xelatex >/dev/null 2>&1; then
  mkdir -p "$SUB_DIR"
  echo "PDF export skipped: xelatex not found on PATH." > "$NOTE_FILE"
  echo "PDF export skipped: xelatex not found. Wrote $NOTE_FILE"
  exit 0
fi

pandoc "$PAPER_MD" \
  --metadata-file "$META_FILE" \
  --from markdown \
  --to pdf \
  --citeproc \
  --bibliography "$BIB_FILE" \
  --template "$TEMPLATE_FILE" \
  --resource-path "$ROOT_DIR:$SUB_DIR" \
  --pdf-engine=xelatex \
  -o "$SUB_DIR/paper.pdf"

echo "PDF exported: $SUB_DIR/paper.pdf"
