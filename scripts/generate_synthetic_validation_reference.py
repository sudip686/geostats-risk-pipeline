from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic synthetic validation reference")
    parser.add_argument("--n", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--out", default=str(ROOT / "submission_ready" / "supplement" / "synthetic_validation_reference.csv"))
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # Spatial extents aligned to published grid envelope for reproducible diagnostic use.
    x = rng.uniform(475500, 476800, args.n)
    y = rng.uniform(9465400, 9470000, args.n)
    z = rng.uniform(570, 840, args.n)

    # Deterministic pseudo-geology grade field (non-project data).
    trend = 4.0 + 0.0045 * (z - 700.0)
    wave = 0.7 * np.sin((x - 475500.0) / 180.0) + 0.5 * np.cos((y - 9465400.0) / 240.0)
    noise = rng.normal(0.0, 0.9, args.n)
    tgc = np.clip(trend + wave + noise, 0.0, 15.0)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"X": x, "Y": y, "Z": z, "TGC_%": tgc})
    df.to_csv(out, index=False)

    meta = {
        "generator": "synthetic_validation_reference",
        "seed": args.seed,
        "n": args.n,
        "formula": "TGC_% = clip(4.0 + 0.0045*(Z-700) + 0.7*sin((X-475500)/180) + 0.5*cos((Y-9465400)/240) + N(0,0.9), 0, 15)",
        "note": "Synthetic deterministic dataset for reproducibility checks only; not project assay data.",
    }
    (out.with_suffix(".meta.json")).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
