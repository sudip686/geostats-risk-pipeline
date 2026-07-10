#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "paper.docx",
    "paper.md",
    "paper_body.md",
    "tables_final.md",
    "figure_captions_final.md",
    "Table.docx",
    "Fig.docx",
    "references.bib",
    "Title page.docx",
    "Covering Letter-MS.docx",
    "Authors Statement.docx",
    "Conflict of interest.docx",
    "Declaration of interest statement.docx",
    "SOURCE_OF_TRUTH.md",
    "source_of_truth.submission.json",
    "submission_package_final_clean.zip",
]

MANUSCRIPT_PRIORITY = [
    "paper.docx",
    "sudip_manuscript.docx",
    "SG Manuscript.docx",
    "SG Manuscript (1).docx",
]

ZIP_FALLBACKS = [ROOT / "internal" / "internal_package_final.zip"]


def find_by_name(name: str, search_roots: Iterable[Path]) -> Path | None:
    target = name.lower()
    for root in search_roots:
        if not root.exists():
            continue
        candidates = sorted(p for p in root.rglob("*") if p.is_file() and p.name.lower() == target)
        if candidates:
            return candidates[0]
    return None


def find_in_zip(name: str, zip_paths: Iterable[Path]) -> tuple[Path, str] | None:
    target = name.lower()
    for zpath in zip_paths:
        if not zpath.exists():
            continue
        with zipfile.ZipFile(zpath, "r") as zf:
            for member in zf.namelist():
                if Path(member).name.lower() == target:
                    return zpath, member
    return None


def copy_member(zip_path: Path, member: str, dst: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open(member, "r") as src, dst.open("wb") as out:
            shutil.copyfileobj(src, out)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild a clean submission folder with fixed required files and canonical manuscript naming."
    )
    parser.add_argument("--source", default="submission", help="Primary source folder (default: submission)")
    parser.add_argument("--out", default="submission", help="Output folder (default: submission)")
    parser.add_argument(
        "--prune-duplicates",
        action="store_true",
        help="Remove duplicate copies of required filenames outside output folder",
    )
    args = parser.parse_args()

    source = (ROOT / args.source).resolve()
    out = (ROOT / args.out).resolve()
    stage = out.parent / (out.name + ".tmp_rebuild")

    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True, exist_ok=True)

    search_roots = [source, ROOT]

    report = {
        "source": str(source.relative_to(ROOT)) if source.exists() else str(source),
        "output": str(out.relative_to(ROOT)),
        "copied": [],
        "missing": [],
        "canonical_manuscript": "paper.docx",
        "canonical_source": None,
    }

    manuscript_done = False
    for mname in MANUSCRIPT_PRIORITY:
        src = find_by_name(mname, search_roots)
        if src is not None:
            shutil.copy2(src, stage / "paper.docx")
            report["canonical_source"] = str(src.relative_to(ROOT))
            report["copied"].append({"name": "paper.docx", "from": report["canonical_source"]})
            manuscript_done = True
            break
        zhit = find_in_zip(mname, ZIP_FALLBACKS)
        if zhit is not None:
            zpath, member = zhit
            copy_member(zpath, member, stage / "paper.docx")
            report["canonical_source"] = f"{zpath.relative_to(ROOT)}::{member}"
            report["copied"].append({"name": "paper.docx", "from": report["canonical_source"]})
            manuscript_done = True
            break

    if not manuscript_done:
        report["missing"].append("paper.docx (or manuscript aliases)")

    for name in REQUIRED_FILES:
        if name == "paper.docx":
            continue
        src = find_by_name(name, search_roots)
        if src is not None:
            shutil.copy2(src, stage / name)
            report["copied"].append({"name": name, "from": str(src.relative_to(ROOT))})
            continue

        zhit = find_in_zip(name, ZIP_FALLBACKS)
        if zhit is not None:
            zpath, member = zhit
            copy_member(zpath, member, stage / name)
            report["copied"].append({"name": name, "from": f"{zpath.relative_to(ROOT)}::{member}"})
            continue

        report["missing"].append(name)

    # Ensure no extras in staged output
    for file in stage.iterdir():
        if file.is_file() and file.name not in REQUIRED_FILES:
            file.unlink()

    report_dir = ROOT / "build"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "submission_rebuild_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    if report["missing"]:
        shutil.rmtree(stage, ignore_errors=True)
        missing = ", ".join(report["missing"])
        raise SystemExit(f"Rebuild blocked. Missing required files: {missing}")

    if out.exists():
        shutil.rmtree(out)
    stage.replace(out)

    if args.prune_duplicates:
        protected = {str((out / n).resolve()) for n in REQUIRED_FILES}
        required_plus_alias = set(REQUIRED_FILES + MANUSCRIPT_PRIORITY)
        for path in ROOT.rglob("*"):
            if not path.is_file() or path.name not in required_plus_alias:
                continue
            resolved = str(path.resolve())
            if resolved in protected:
                continue
            try:
                path.unlink()
            except OSError:
                pass

    print("Rebuild complete:", out)


if __name__ == "__main__":
    main()
