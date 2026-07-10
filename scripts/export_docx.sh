#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUB_DIR="$ROOT_DIR/submission"
PAPER_MD="$SUB_DIR/paper.md"

if ! command -v pandoc >/dev/null 2>&1; then
  if [ -x "/c/Users/SUDIPTA CHANDA/AppData/Local/Pandoc/pandoc.exe" ]; then
    export PATH="$PATH:/c/Users/SUDIPTA CHANDA/AppData/Local/Pandoc"
  fi
fi

if ! command -v pandoc >/dev/null 2>&1; then
  echo "pandoc not found. Install pandoc to export DOCX."
  exit 1
fi

if [ ! -f "$PAPER_MD" ]; then
  echo "Missing $PAPER_MD. Run: python scripts/build_paper.py"
  exit 1
fi

BIB_FILE="$ROOT_DIR/references.bib"
META_FILE="$SUB_DIR/paper.yaml"
if [ ! -f "$BIB_FILE" ]; then
  BIB_FILE="$ROOT_DIR/paper/references.bib"
fi

if [ ! -f "$BIB_FILE" ]; then
  echo "references.bib not found at root or paper/."
  exit 1
fi

pandoc "$PAPER_MD" \
  --metadata-file "$META_FILE" \
  --from markdown \
  --to docx \
  --citeproc \
  --bibliography "$BIB_FILE" \
  --resource-path "$ROOT_DIR:$SUB_DIR" \
  -o "$SUB_DIR/paper.docx"

echo "DOCX exported: $SUB_DIR/paper.docx"
