"""Internal validation wrapper for MODEL_OK vs SGS comparisons."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from src.utils.io import load_config


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "Tanga_MRE_2026-01-06 1" / "OneDrive_2026-01-06" / "Export Final" / "04 BM" / "CSV" / "MODEL_OK.csv"


def run(config_path: str = "config/main_config.yaml", output_dir: str = "outputs") -> dict:
    cfg = load_config(config_path)
    iv_cfg = cfg.get("internal_validation", {})

    if not iv_cfg.get("enabled", True):
        return {"enabled": False, "status": "skipped"}

    model_csv = Path(iv_cfg.get("model_csv", str(DEFAULT_MODEL)))
    script_path = ROOT / "src" / "internal_validation_block_model.py"

    cmd = [
        sys.executable,
        str(script_path),
        "--model-csv",
        str(model_csv),
        "--outputs-dir",
        str(Path(output_dir)),
        "--config",
        str(Path(config_path)),
    ]
    try:
        subprocess.run(cmd, check=True)
        return {"enabled": True, "status": "completed", "model_csv": str(model_csv)}
    except subprocess.CalledProcessError as exc:
        return {
            "enabled": True,
            "status": "failed",
            "model_csv": str(model_csv),
            "error": str(exc),
        }

