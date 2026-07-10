from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "main_config.yaml"
DEFAULT_RUN_DIR = ROOT / "output" / "a3_categorical_25_50_nr100"
DEFAULT_WORKBOOK_DIR = ROOT / "build" / "notebooklm_canonical_review_pack"
DEFAULT_NOTEBOOK_TITLE = "N6_JAES_Reviewer_Fixes_2m_Grid"


def run_cmd(cmd: list[str], env: dict[str, str] | None = None) -> None:
    proc = subprocess.run(cmd, cwd=ROOT, env=env, check=False)
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def run_workflow(config: Path, run_dir: Path, smoke: bool) -> None:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if smoke:
        env["CI"] = "true"
    run_cmd([sys.executable, "-m", "src.run_all", "--config", str(config), "--output", str(run_dir)], env=env)


def run_diagnostics(run_dir: Path, config: Path) -> dict[str, object]:
    sys.path.insert(0, str(ROOT))
    from src.nugget_decomposition import run as run_nugget
    from src.reviewer_upgrade_pack import run as run_review_pack
    from src.utils.io import load_config

    cfg = load_config(str(config))
    return {
        "reviewer_upgrade_pack": run_review_pack(output_dir=str(run_dir), config=cfg),
        "nugget_decomposition": run_nugget(output_dir=str(run_dir), config=cfg),
    }


def build_workbook(run_dir: Path, workbook_dir: Path) -> None:
    run_cmd(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_jaes_vertical2m_workbook.py"),
            "--run-dir",
            str(run_dir),
            "--out-dir",
            str(workbook_dir),
        ]
    )


def notebooklm_available() -> tuple[bool, str]:
    proc = subprocess.run(
        ["notebooklm", "list", "--json"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == 0:
        return True, proc.stdout.strip()
    return False, (proc.stderr or proc.stdout).strip()


def create_notebook_and_add_sources(workbook_dir: Path, notebook_title: str) -> dict[str, object]:
    ok, detail = notebooklm_available()
    if not ok:
        return {"status": "auth_required", "detail": detail}

    create = subprocess.run(
        ["notebooklm", "create", notebook_title, "--json"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if create.returncode != 0:
        return {"status": "create_failed", "detail": (create.stderr or create.stdout).strip()}

    notebook = json.loads(create.stdout)
    notebook_id = notebook.get("id")
    if notebook_id is None and isinstance(notebook.get("notebook"), dict):
        notebook_id = notebook["notebook"].get("id")
    if notebook_id is None:
        return {"status": "create_failed", "detail": f"Unexpected create response: {create.stdout.strip()}"}
    manifest = workbook_dir / "notebooklm_sources_manifest.csv"
    added = []
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            proc = subprocess.run(
                [
                    "notebooklm",
                    "source",
                    "add",
                    row["path"],
                    "-n",
                    notebook_id,
                    "--title",
                    row["title"],
                    "--json",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            added.append(
                {
                    "path": row["path"],
                    "title": row["title"],
                    "returncode": proc.returncode,
                    "stdout": proc.stdout.strip(),
                    "stderr": proc.stderr.strip(),
                }
            )
    result = {"status": "completed", "notebook_id": notebook_id, "notebook_title": notebook_title, "sources": added}
    (workbook_dir / "notebooklm_import_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the JAES 2 m workflow branch, diagnostics, workbook, and NotebookLM import.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--workbook-dir", default=str(DEFAULT_WORKBOOK_DIR))
    parser.add_argument("--notebook-title", default=DEFAULT_NOTEBOOK_TITLE)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument("--skip-diagnostics", action="store_true")
    parser.add_argument("--skip-workbook", action="store_true")
    parser.add_argument("--skip-notebooklm", action="store_true")
    args = parser.parse_args()

    config = Path(args.config)
    run_dir = Path(args.run_dir)
    workbook_dir = Path(args.workbook_dir)
    if not config.is_absolute():
        config = ROOT / config
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    if not workbook_dir.is_absolute():
        workbook_dir = ROOT / workbook_dir

    status: dict[str, object] = {
        "config": str(config),
        "run_dir": str(run_dir),
        "workbook_dir": str(workbook_dir),
        "smoke": bool(args.smoke),
    }

    if not args.skip_run:
        run_workflow(config=config, run_dir=run_dir, smoke=bool(args.smoke))
        status["workflow"] = "completed"
    else:
        status["workflow"] = "skipped"

    if not args.skip_diagnostics:
        status["diagnostics"] = run_diagnostics(run_dir=run_dir, config=config)
    else:
        status["diagnostics"] = "skipped"

    if not args.skip_workbook:
        build_workbook(run_dir=run_dir, workbook_dir=workbook_dir)
        status["workbook"] = "completed"
    else:
        status["workbook"] = "skipped"

    if not args.skip_notebooklm:
        status["notebooklm"] = create_notebook_and_add_sources(workbook_dir=workbook_dir, notebook_title=args.notebook_title)
    else:
        status["notebooklm"] = "skipped"

    (workbook_dir / "pipeline_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
