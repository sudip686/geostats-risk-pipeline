from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print("RUN", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Single strict regeneration path with drift guard.")
    parser.add_argument("--run-dir", default=str(ROOT / "output" / "review_closure" / "baseline_combined"))
    parser.add_argument("--config", default=str(ROOT / "output" / "review_closure" / "configs" / "baseline_combined.yaml"))
    parser.add_argument("--audit-report", default=str(ROOT / "review" / "STRICT_DRIFT_REPORT_latest.md"))
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()

    run(
        [
            "python",
            "scripts/generate_cv_evidence.py",
            "--config",
            args.config,
            "--output-dir",
            args.run_dir,
            "--sync-submission-ready",
            "--sync-final-zip",
        ]
    )

    if not args.skip_build:
        run(["python", "scripts/build_submission_package.py", "--run-dir", args.run_dir, "--strict"])

    run(
        [
            "python",
            "scripts/strict_regen_guard.py",
            "audit",
            "--submission-dir",
            str(ROOT / "submission_final_JAES"),
            "--manifest",
            str(ROOT / "review" / "STRICT_BASELINE_MANIFEST.json"),
            "--policy",
            str(ROOT / "review" / "STRICT_ALLOWED_DELTAS.yaml"),
            "--report",
            args.audit_report,
        ]
    )

    print("Strict regeneration completed.")


if __name__ == "__main__":
    main()
