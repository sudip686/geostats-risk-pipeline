from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing watch status file: {path}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform.startswith("win"):
        proc = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        line = (proc.stdout or "").strip()
        return bool(line) and not line.startswith("INFO:")
    try:
        import os

        os.kill(pid, 0)
        return True
    except OSError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Wait for a workflow PID to exit, then run the canonical post-run pack.")
    parser.add_argument("--status-file", required=True)
    parser.add_argument("--interval-sec", type=int, default=60)
    parser.add_argument("--top-n", type=int, default=50)
    args = parser.parse_args()

    status_file = Path(args.status_file)
    if not status_file.is_absolute():
        status_file = ROOT / status_file
    status = load_json(status_file)

    pid = int(status.get("pid", 0))
    run_dir = Path(status.get("run_dir", ""))
    config = Path(status.get("config", ROOT / "config" / "main_config.yaml"))
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    if not config.is_absolute():
        config = ROOT / config

    state_path = ROOT / "build" / f"{status_file.stem}_postrun_watch.json"
    log_path = ROOT / "build" / f"{status_file.stem}_postrun_watch.log"
    payload = {
        "status_file": str(status_file),
        "pid": pid,
        "run_dir": str(run_dir),
        "config": str(config),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "state": "waiting",
    }
    write_json(state_path, payload)

    while pid_exists(pid):
        payload["last_checked_at"] = datetime.now().isoformat(timespec="seconds")
        write_json(state_path, payload)
        time.sleep(max(int(args.interval_sec), 5))

    payload["state"] = "running_postrun_pack"
    payload["workflow_exited_at"] = datetime.now().isoformat(timespec="seconds")
    write_json(state_path, payload)

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_canonical_postrun_pack.py"),
        "--config",
        str(config),
        "--run-dir",
        str(run_dir),
        "--top-n",
        str(int(args.top_n)),
    ]
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    log_text = "=== COMMAND ===\n" + " ".join(cmd) + "\n\n=== STDOUT ===\n" + (proc.stdout or "") + "\n=== STDERR ===\n" + (proc.stderr or "")
    log_path.write_text(log_text, encoding="utf-8")

    payload["state"] = "completed" if proc.returncode == 0 else "failed"
    payload["finished_at"] = datetime.now().isoformat(timespec="seconds")
    payload["postrun_command"] = cmd
    payload["postrun_exit_code"] = proc.returncode
    payload["log_path"] = str(log_path)
    write_json(state_path, payload)

    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
