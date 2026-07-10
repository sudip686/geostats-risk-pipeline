from __future__ import annotations

import logging
import os
import subprocess
import shutil
import time
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR_FALLBACK = "output/a3_geology_aligned_250_200_20_nr100"
CONFIG = "config/main_config.yaml"
MODEL_OK = r"Tanga_MRE_2026-01-06 1\OneDrive_2026-01-06\Export Final\04 BM\CSV\MODEL_OK.csv"
CANONICAL_N_REAL = 100
CANONICAL_SEARCH_RADIUS = [250, 200, 20]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("pipeline")


def resolve_run_dir() -> str:
    def _is_complete_run_dir(p: Path) -> bool:
        required = [
            p / "domain_data.csv",
            p / "nst_data.csv",
            p / "grids" / "sgs_reals.npy",
            p / "grids" / "sgs_meta.json",
            p / "tables" / "risked_tonnage.csv",
            p / "tables" / "validation_metrics.json",
        ]
        return all(x.exists() for x in required)

    def _to_rel(p: Path) -> str:
        return str(p.relative_to(ROOT))

    best_summary = ROOT / "output" / "best_summary.json"
    if best_summary.exists():
        try:
            data = json.loads(best_summary.read_text(encoding="utf-8"))
            out_dir = data.get("best_output_dir")
            if out_dir:
                p = Path(out_dir)
                if p.exists() and _is_complete_run_dir(p):
                    return _to_rel(p)
                alt = Path(str(out_dir).replace("outputs_fit_tuning", "output"))
                if alt.exists() and _is_complete_run_dir(alt):
                    return _to_rel(alt)
        except Exception:
            pass

    # Choose newest complete run directory under output/ when best_summary is stale/incomplete.
    output_root = ROOT / "output"
    if output_root.exists():
        candidates = []
        for d in output_root.iterdir():
            if d.is_dir() and _is_complete_run_dir(d):
                candidates.append(d)
        if candidates:
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return _to_rel(candidates[0])

    # Final fallback remains deterministic.
    fallback = ROOT / RUN_DIR_FALLBACK
    if fallback.exists() and _is_complete_run_dir(fallback):
        return RUN_DIR_FALLBACK

    return RUN_DIR_FALLBACK


def _format_cmd(cmd: list[str]) -> str:
    return " ".join(cmd)


def _is_canonical_submission_run(run_dir_rel: str) -> tuple[bool, str]:
    run_dir = ROOT / run_dir_rel
    meta_path = run_dir / "sgs_meta.json"
    if not meta_path.exists():
        return False, f"missing {meta_path}"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"failed to read {meta_path}: {exc}"

    cfg = meta.get("config", {})
    sim = cfg.get("simulation", {})
    tuning = cfg.get("variogram", {}).get("tuning", {})
    trend = cfg.get("trend", {})
    calib = cfg.get("calibration", {})

    checks = {
        f"n_real={CANONICAL_N_REAL}": int(sim.get("n_real", 0)) == CANONICAL_N_REAL,
        f"search_radius={CANONICAL_SEARCH_RADIUS}": list(sim.get("search_radius_m", [])) == CANONICAL_SEARCH_RADIUS,
        "tuning_enabled": bool(tuning.get("enabled", False)),
        "trend_enabled": bool(trend.get("enabled", False)),
        "calibration_disabled": not bool(calib.get("enabled", False)),
    }
    failed = [k for k, ok in checks.items() if not ok]
    if failed:
        return False, ", ".join(failed)
    return True, "ok"


def run(step: str, cmd: list[str]) -> None:
    start = time.time()
    logger.info("START %s", step)
    logger.info("CMD   %s", _format_cmd(cmd))
    logger.info("CWD   %s", ROOT)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=env,
    )

    last_heartbeat = time.time()
    heartbeat_sec = 30
    while True:
        rc = proc.poll()
        if rc is not None:
            break
        now = time.time()
        if now - last_heartbeat >= heartbeat_sec:
            logger.info("[%s] still running... elapsed %.1fs", step, now - start)
            last_heartbeat = now
        time.sleep(1)

    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)
    logger.info("DONE  %s (%.1fs)", step, time.time() - start)


def main() -> None:
    total_start = time.time()
    run_dir = RUN_DIR_FALLBACK
    logger.info("Pipeline started")
    run_dir_abs = ROOT / run_dir
    if run_dir_abs.exists():
        logger.info("Cleaning canonical run directory for fresh generation: %s", run_dir_abs)
        shutil.rmtree(run_dir_abs)
    run_dir_abs.mkdir(parents=True, exist_ok=True)
    run("0/3 Full canonical run", [
        "python",
        "-m",
        "src.run_all",
        "--config",
        CONFIG,
        "--output",
        run_dir,
    ])
    canonical_ok, reason = _is_canonical_submission_run(run_dir)
    if not canonical_ok:
        raise RuntimeError(f"Fresh run failed canonical checks: {reason}")
    logger.info("Active run directory: %s", run_dir)
    run("1/3 Variogram reproduction", [
        "python",
        "src/variogram_reproduction.py",
        "--config", CONFIG,
        "--outputs", run_dir,
        "--n-real-eval",
        "20",
        "--max-grid-samples",
        "2500",
    ])
    run("2/3 Internal validation (MODEL_OK vs SGS + assay/domain)", [
        "python",
        "src/internal_validation_block_model.py",
        "--model-csv", MODEL_OK,
        "--outputs-dir", run_dir,
        "--config", CONFIG,
    ])
    run("3/3 Build submission package", ["python", "scripts/build_submission_package.py", "--run-dir", run_dir, "--strict"])
    logger.info("Pipeline complete (%.1fs)", time.time() - total_start)
    logger.info("Internal report: %s", ROOT / run_dir / "internal_validation" / "MODEL_OK_vs_SGS_similarity_report.md")


if __name__ == "__main__":
    main()

