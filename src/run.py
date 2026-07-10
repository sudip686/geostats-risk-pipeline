"""Compatibility entrypoint that forwards to the full run_all workflow."""

from __future__ import annotations

import argparse
import sys

from src.run_all import run_full_workflow


def main() -> None:
    parser = argparse.ArgumentParser(description="Graphite SGS workflow (compat entrypoint)")
    parser.add_argument("--config", default="config/main_config.yaml", help="Config file path")
    parser.add_argument("--output", default="outputs", help="Output directory")
    args = parser.parse_args()

    ok = run_full_workflow(config_path=args.config, output_dir=args.output)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
