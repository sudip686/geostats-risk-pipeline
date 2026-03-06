from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "submission"
ZIP_PATH = SUBMISSION / "submission_package.zip"


def make_zip() -> Path:
    SUBMISSION.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in SUBMISSION.rglob("*"):
            if path == ZIP_PATH or not path.is_file():
                continue
            zf.write(path, path.relative_to(SUBMISSION))
    return ZIP_PATH


if __name__ == "__main__":
    out = make_zip()
    print(f"Created {out}")
