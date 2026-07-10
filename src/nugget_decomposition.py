"""Compare raw-grade and log-grade nugget behavior."""

from __future__ import annotations

import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .variography import estimate_variogram, fit_variogram_model

logger = logging.getLogger(__name__)


def _load_domain_data(output_dir: str) -> pd.DataFrame:
    path = os.path.join(output_dir, 'domain_data.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing domain_data.csv in {output_dir}")
    return pd.read_csv(path)


def _fit_profile(coords, values, config, label):
    vario_cfg = (config or {}).get('variogram', {}) if config else {}
    bins, gamma = estimate_variogram(
        coords,
        values,
        n_lags=int(vario_cfg.get('n_lags', 10)),
        max_dist=float(vario_cfg.get('max_distance_m', 500)),
    )
    model = fit_variogram_model(
        bins,
        gamma,
        model_type=(vario_cfg.get('model_types') or ['exponential'])[0],
    )
    total_sill = float(model.nugget + model.var)
    nugget_ratio = float(model.nugget / total_sill) if total_sill > 0 else np.nan
    return {
        'transform': label,
        'bins': bins,
        'gamma': gamma,
        'model': model,
        'nugget': float(model.nugget),
        'structured_sill': float(model.var),
        'total_sill': total_sill,
        'nugget_ratio': nugget_ratio,
        'range_m': float(model.len_scale),
    }


def _plot_profiles(raw_profile, log_profile, out_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=False)
    for ax, profile, title in zip(axes, [raw_profile, log_profile], ['Raw Grade', 'log(1+Grade)']):
        valid = ~np.isnan(profile['gamma'])
        ax.plot(profile['bins'][valid], profile['gamma'][valid], 'o', color='#1f77b4', label='Experimental')
        x_fit = np.linspace(0.1, max(profile['bins'][valid].max(), 1.0), 200)
        model = profile['model']
        y_fit = model.nugget + model.var * (1 - model.cor(x_fit / model.len_scale))
        ax.plot(x_fit, y_fit, '-', color='#d1495b', linewidth=2, label='Fit')
        ax.set_title(title)
        ax.set_xlabel('Distance (m)')
        ax.set_ylabel('Gamma')
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def run(output_dir: str = 'outputs', config: dict | None = None) -> dict:
    figures_dir = os.path.join(output_dir, 'figures')
    tables_dir = os.path.join(output_dir, 'tables')
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    df = _load_domain_data(output_dir)
    coords = (df['x'].to_numpy(dtype=float), df['y'].to_numpy(dtype=float), df['z'].to_numpy(dtype=float))
    raw_values = df['tgc_pct'].to_numpy(dtype=float)
    log_values = np.log1p(np.clip(raw_values, a_min=0.0, a_max=None))

    raw_profile = _fit_profile(coords, raw_values, config, 'raw_grade')
    log_profile = _fit_profile(coords, log_values, config, 'log1p_grade')

    summary = pd.DataFrame(
        [
            {k: v for k, v in raw_profile.items() if k not in {'bins', 'gamma', 'model'}},
            {k: v for k, v in log_profile.items() if k not in {'bins', 'gamma', 'model'}},
        ]
    )
    summary_path = os.path.join(tables_dir, 'nugget_decomposition.csv')
    summary.to_csv(summary_path, index=False)

    figure_path = os.path.join(figures_dir, 'nugget_decomposition.png')
    _plot_profiles(raw_profile, log_profile, figure_path)

    meta = {
        'raw_minus_log_nugget_ratio': float(raw_profile['nugget_ratio'] - log_profile['nugget_ratio']),
        'raw_nugget_ratio': float(raw_profile['nugget_ratio']),
        'log_nugget_ratio': float(log_profile['nugget_ratio']),
    }
    with open(os.path.join(tables_dir, 'nugget_decomposition_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    logger.info("Saved nugget decomposition outputs to %s", output_dir)
    return {'summary': summary_path, 'figure': figure_path, 'meta': meta}


if __name__ == '__main__':
    run()
