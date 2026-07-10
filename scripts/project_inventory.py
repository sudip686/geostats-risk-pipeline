from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "build" / "project_inventory.json"

RUNTIME_REQUIRED = {
    "config",
    "data",
    "scripts",
    "src",
    "outputs_fit_tuning",
    "Tanga_MRE_2026-01-06 1",
}

DELIVERABLES = {"submission", "internal", "repo", "paper"}

ARCHIVE_ONLY = {"_archive_20260307"}


def classify_top_level(path: Path) -> str:
    name = path.name
    if name in RUNTIME_REQUIRED:
        return "runtime_required"
    if name in DELIVERABLES:
        return "deliverable"
    if name.startswith("_archive_") or name in ARCHIVE_ONLY:
        return "archive_only"
    return "reference_or_misc"


def main() -> None:
    entries = []
    for p in sorted(ROOT.iterdir(), key=lambda x: x.name.lower()):
        if p.name in {".git", ".venv"}:
            continue
        kind = "dir" if p.is_dir() else "file"
        size = 0
        if p.is_file():
            size = p.stat().st_size
        else:
            for f in p.rglob("*"):
                if f.is_file():
                    size += f.stat().st_size
        entries.append(
            {
                "name": p.name,
                "kind": kind,
                "size_bytes": size,
                "class": classify_top_level(p),
            }
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(entries, indent=2), encoding="utf-8")

    counts: dict[str, int] = {}
    for e in entries:
        counts[e["class"]] = counts.get(e["class"], 0) + 1
    print("Inventory written:", OUT_PATH)
    print("Class counts:", counts)


if __name__ == "__main__":
    main()
