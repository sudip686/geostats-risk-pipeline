from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "stale_project_minerals.yaml"
DEFAULT_RUN_DIR = ROOT / "output" / "minerals_point_support_nr100"
DEFAULT_PACKAGE_DIR = ROOT / "submission_minerals_ready"


def run_cmd(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, cwd=ROOT, check=False)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Minerals SGS workflow and build the submission package.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--package-dir", default=str(DEFAULT_PACKAGE_DIR))
    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()

    config = Path(args.config)
    run_dir = Path(args.run_dir)
    package_dir = Path(args.package_dir)

    if not config.is_absolute():
        config = ROOT / config
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    if not package_dir.is_absolute():
        package_dir = ROOT / package_dir

    if not args.skip_run:
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_cmd(
            [
                sys.executable,
                "-m",
                "src.run_all",
                "--config",
                str(config),
                "--output",
                str(run_dir),
            ]
        )

    if not args.skip_build:
        run_cmd(
            [
                sys.executable,
                str(ROOT / "scripts" / "build_minerals_package.py"),
                "--run-dir",
                str(run_dir),
                "--output-dir",
                str(package_dir),
            ]
        )


if __name__ == "__main__":
    main()
