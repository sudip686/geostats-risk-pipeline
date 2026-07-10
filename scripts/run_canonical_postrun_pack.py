from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reviewer_upgrade_pack import run as run_reviewer_upgrade_pack
from src.utils.io import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the canonical reviewer-facing post-run diagnostics pack.")
    parser.add_argument("--config", default=str(ROOT / "config" / "main_config.yaml"))
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--top-n", type=int, default=50)
    args = parser.parse_args()

    config = Path(args.config)
    run_dir = Path(args.run_dir)
    if not config.is_absolute():
        config = ROOT / config
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir

    cfg = load_config(str(config))
    result = run_reviewer_upgrade_pack(output_dir=str(run_dir), config=cfg, top_n=int(args.top_n))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

