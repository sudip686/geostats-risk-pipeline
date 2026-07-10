from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cross_validation import run as run_cv
from src.normal_score import NormalScoreTransform


DEFAULT_CONFIG = ROOT / "output" / "review_closure" / "configs" / "baseline_combined.yaml"
DEFAULT_OUTPUT_DIR = ROOT / "output" / "review_closure" / "baseline_combined"


def ensure_tgc_ns_column(output_dir: Path) -> None:
    decl = output_dir / "declustered.csv"
    if not decl.exists():
        raise FileNotFoundError(f"Missing declustered data: {decl}")
    df = pd.read_csv(decl)
    if "tgc_ns" in df.columns:
        return
    values = df["tgc_pct"].to_numpy(dtype=float)
    weights = df["decluster_weight"].to_numpy(dtype=float) if "decluster_weight" in df.columns else None
    nst = NormalScoreTransform().fit(values, weights=weights)
    df["tgc_ns"] = nst.transform(values)
    df.to_csv(decl, index=False)


def generate(output_dir: Path, config_path: Path, seed: int = 42) -> None:
    ensure_tgc_ns_column(output_dir)
    run_cv(
        config_path=str(config_path),
        output_dir=str(output_dir),
        max_samples=300,
        k_folds=5,
        seed=seed,
        fold_mode="random",
        output_name="cross_validation_300.json",
    )
    run_cv(
        config_path=str(config_path),
        output_dir=str(output_dir),
        max_samples=600,
        k_folds=5,
        seed=seed,
        fold_mode="random",
        output_name="cross_validation_600.json",
    )
    run_cv(
        config_path=str(config_path),
        output_dir=str(output_dir),
        max_samples=300,
        k_folds=5,
        seed=seed,
        fold_mode="blocked",
        block_size_xy=500.0,
        output_name="cross_validation_blocked_300.json",
    )


def verify_modes(output_dir: Path) -> None:
    tables = output_dir / "tables"
    cv300 = json.loads((tables / "cross_validation_300.json").read_text(encoding="utf-8"))
    cv600 = json.loads((tables / "cross_validation_600.json").read_text(encoding="utf-8"))
    cvb = json.loads((tables / "cross_validation_blocked_300.json").read_text(encoding="utf-8"))
    if cv300.get("fold_mode") != "random" or cv600.get("fold_mode") != "random":
        raise RuntimeError("Random CV evidence generation failed: fold_mode is not random for 300/600 files.")
    if cvb.get("fold_mode") != "blocked":
        raise RuntimeError("Blocked CV evidence generation failed: fold_mode is not blocked for blocked_300 file.")
    if cv300 == cvb or cv600 == cvb:
        raise RuntimeError("Random CV evidence appears identical to blocked CV evidence.")


def sync_to_submission_ready(output_dir: Path) -> None:
    sup = ROOT / "submission_ready" / "supplement"
    sup.mkdir(parents=True, exist_ok=True)
    for name in ["cross_validation_300.json", "cross_validation_600.json", "cross_validation_blocked_300.json"]:
        shutil.copy2(output_dir / "tables" / name, sup / name)


def sync_to_final_zip(output_dir: Path) -> None:
    zip_path = ROOT / "submission_final_JAES" / "Supplementary_Data_S2.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"Missing final supplementary zip: {zip_path}")
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp)
        for name in ["cross_validation_300.json", "cross_validation_600.json", "cross_validation_blocked_300.json"]:
            shutil.copy2(output_dir / "tables" / name, tmp / name)
        rebuilt = tmp / "rebuilt.zip"
        with zipfile.ZipFile(rebuilt, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(tmp.rglob("*")):
                if p == rebuilt or p.is_dir():
                    continue
                zf.write(p, p.relative_to(tmp).as_posix())
        shutil.copy2(rebuilt, zip_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and sync auditable random-vs-blocked CV evidence files.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sync-submission-ready", action="store_true")
    parser.add_argument("--sync-final-zip", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    output_dir = Path(args.output_dir)
    generate(output_dir=output_dir, config_path=config_path, seed=args.seed)
    verify_modes(output_dir=output_dir)
    if args.sync_submission_ready:
        sync_to_submission_ready(output_dir=output_dir)
    if args.sync_final_zip:
        sync_to_final_zip(output_dir=output_dir)
    print("CV evidence generated and verified.")


if __name__ == "__main__":
    main()
