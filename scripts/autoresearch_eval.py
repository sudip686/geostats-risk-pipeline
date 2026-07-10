from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "build" / "autoresearch_eval_latest.json"
LOG_PATH = ROOT / "build" / "autoresearch_eval_latest.log"


def run_command(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    output = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, output


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def precheck_path(sub_dir: str) -> Path:
    if sub_dir == "submission":
        work_report = ROOT / "build" / "submission_work" / "preflight_report.json"
        if work_report.exists():
            return work_report
    return ROOT / sub_dir / "preflight_report.json"


def detect_default_run_dir(sub_dir: str) -> Path | None:
    report = load_json(precheck_path(sub_dir))
    value = report.get("run_dir")
    if isinstance(value, str) and value.strip():
        candidate = Path(value)
        if candidate.exists():
            return candidate
    return None


def parse_preflight(report: dict[str, Any]) -> dict[str, Any]:
    checks = report.get("checks", {}) if isinstance(report, dict) else {}
    missing = list(checks.get("missing_required_items", []) or [])
    content = list(checks.get("content_issues", []) or [])
    visual = list(checks.get("visual_issues", []) or [])
    final_submission = list(checks.get("final_submission_issues", []) or [])
    issue_count = len(missing) + len(content) + len(visual) + len(final_submission)
    status = str(report.get("status", "unknown") or "unknown")
    return {
        "status": status,
        "issue_count": issue_count,
        "missing_required_items": missing,
        "content_issues": content,
        "visual_issues": visual,
        "final_submission_issues": final_submission,
    }


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    build_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "build_submission_package.py"),
        "--strict",
    ]
    if args.independent:
        build_cmd.append("--independent")
    if args.run_dir is not None:
        build_cmd.extend(["--run-dir", str(args.run_dir)])

    build_exit_code, build_output = run_command(build_cmd)
    preflight_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "submission_preflight.py"),
        "--sub-dir",
        args.sub_dir,
    ]
    preflight_exit_code, preflight_output = run_command(preflight_cmd)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(
        "=== BUILD ===\n"
        + build_output
        + "\n=== PREFLIGHT ===\n"
        + preflight_output,
        encoding="utf-8",
    )

    preflight = load_json(precheck_path(args.sub_dir))
    parsed = parse_preflight(preflight)

    # Penalize command-level failures that do not surface as preflight issues.
    overall_exit_code = build_exit_code if build_exit_code != 0 else preflight_exit_code
    if overall_exit_code != 0 and parsed["issue_count"] == 0:
        parsed["issue_count"] = 1
        if parsed["status"] == "pass":
            parsed["status"] = "fail"

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "objective": "minimize_issue_count",
        "build_command": build_cmd,
        "build_exit_code": build_exit_code,
        "preflight_command": preflight_cmd,
        "preflight_exit_code": preflight_exit_code,
        "command_exit_code": overall_exit_code,
        "log_path": str(LOG_PATH),
        "summary_source": str(precheck_path(args.sub_dir)),
        "package_profile": "independent" if args.independent else "mme",
        **parsed,
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one strict autoresearch evaluation for the Tanga_New project.")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--independent", action="store_true")
    args = parser.parse_args()
    args.sub_dir = "submission_ready_independent" if args.independent else "submission"

    if args.run_dir:
        args.run_dir = Path(args.run_dir)
    else:
        args.run_dir = detect_default_run_dir(args.sub_dir)

    summary = build_summary(args)
    print(json.dumps(summary, indent=2))

    if int(summary.get("command_exit_code", 1)) != 0:
        raise SystemExit(1)
    if str(summary.get("status", "")) != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
