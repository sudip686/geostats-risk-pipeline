"""Compute spatial confidence-gradient summaries from reporting-support percentiles."""

from __future__ import annotations

import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _load_reporting_meta(output_dir: str) -> dict:
    meta_path = os.path.join(output_dir, 'grids', 'sgs_reporting_meta.json')
    if not os.path.exists(meta_path):
        meta_path = os.path.join(output_dir, 'grids', 'sgs_meta.json')
    with open(meta_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def run(output_dir: str = 'outputs', top_n: int = 50) -> dict:
    grids_dir = os.path.join(output_dir, 'grids')
    figures_dir = os.path.join(output_dir, 'figures')
    tables_dir = os.path.join(output_dir, 'tables')
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    p10 = np.load(os.path.join(grids_dir, 'p10_grid.npy'))
    p50 = np.load(os.path.join(grids_dir, 'p50_grid.npy'))
    p90 = np.load(os.path.join(grids_dir, 'p90_grid.npy'))
    meta = _load_reporting_meta(output_dir)

    x = meta['x_min'] + np.arange(int(meta['nx'])) * float(meta['dx'])
    y = meta['y_min'] + np.arange(int(meta['ny'])) * float(meta['dy'])
    z = meta['z_min'] + np.arange(int(meta['nz'])) * float(meta['dz'])

    aperture = np.divide(
        (p90 - p10),
        p50,
        out=np.full_like(p50, np.nan, dtype=float),
        where=np.abs(p50) > 1e-9,
    ) * 100.0
    spread = p90 - p10

    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    df = pd.DataFrame(
        {
            'x': xx.ravel(),
            'y': yy.ravel(),
            'z': zz.ravel(),
            'p10_grade': p10.ravel(),
            'p50_grade': p50.ravel(),
            'p90_grade': p90.ravel(),
            'risk_aperture_pct': aperture.ravel(),
            'grade_spread': spread.ravel(),
        }
    ).dropna(subset=['risk_aperture_pct'])
    hotspots = df.sort_values('risk_aperture_pct', ascending=False).head(top_n).copy()
    hotspot_path = os.path.join(tables_dir, 'confidence_gradient_hotspots.csv')
    hotspots.to_csv(hotspot_path, index=False)

    aperture_map = np.nanmax(aperture, axis=2)
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    im = ax.imshow(
        aperture_map.T,
        origin='lower',
        aspect='auto',
        extent=[x.min(), x.max(), y.min(), y.max()],
        cmap='viridis',
    )
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Max Risk Aperture (%)')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Spatial Confidence Gradient (max over Z)')
    plt.tight_layout()
    figure_path = os.path.join(figures_dir, 'confidence_gradient_map.png')
    plt.savefig(figure_path, dpi=180)
    plt.close(fig)

    meta_out = {
        'max_risk_aperture_pct': float(np.nanmax(aperture)),
        'median_risk_aperture_pct': float(np.nanmedian(aperture)),
        'hotspot_count': int(len(hotspots)),
    }
    with open(os.path.join(tables_dir, 'confidence_gradient_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta_out, f, indent=2)

    logger.info("Saved confidence-gradient outputs to %s", output_dir)
    return {'hotspots': hotspot_path, 'figure': figure_path, 'meta': meta_out}


if __name__ == '__main__':
    run()
