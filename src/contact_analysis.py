"""Contact analysis for fresh vs. weathered graphitic contacts."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


WEATHERED_PREFIXES = ("SAP", "SAPR")


@dataclass(frozen=True)
class ContactBin:
    label: str
    min_dist: float
    max_dist: float
    midpoint: float


CONTACT_BINS = [
    ContactBin("0-2 m", 0.0, 2.0, 1.0),
    ContactBin("2-5 m", 2.0, 5.0, 3.5),
    ContactBin("5-10 m", 5.0, 10.0, 7.5),
]


def _classify_weathering(lith_code: str) -> str:
    lith = str(lith_code).upper()
    return "weathered" if lith.startswith(WEATHERED_PREFIXES) else "fresh"


def _load_domain_frame(output_dir: str) -> pd.DataFrame:
    for candidate in ("domain_data.csv", "composites.csv"):
        path = os.path.join(output_dir, candidate)
        if os.path.exists(path):
            df = pd.read_csv(path)
            if {'hole_id', 'from_m', 'to_m', 'tgc_pct', 'lith_code'}.issubset(df.columns):
                return df
    raise FileNotFoundError("Expected domain_data.csv or composites.csv with hole_id/from_m/to_m/tgc_pct/lith_code")


def _build_contact_rows(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = df.sort_values(['hole_id', 'from_m']).groupby('hole_id', sort=False)
    for hole_id, hole in grouped:
        hole = hole.copy()
        hole['weathering_class'] = hole['lith_code'].map(_classify_weathering)
        contact_depths = []
        for i in range(len(hole) - 1):
            left = hole.iloc[i]
            right = hole.iloc[i + 1]
            if left['weathering_class'] == right['weathering_class']:
                continue
            if abs(float(left['to_m']) - float(right['from_m'])) > 0.25:
                continue
            contact_depths.append(float(left['to_m']))
        if not contact_depths:
            continue

        for _, row in hole.iterrows():
            mid = 0.5 * (float(row['from_m']) + float(row['to_m']))
            nearest = min(contact_depths, key=lambda d: abs(mid - d))
            signed_distance = mid - nearest
            abs_distance = abs(signed_distance)
            if abs_distance > CONTACT_BINS[-1].max_dist:
                continue
            rows.append(
                {
                    'hole_id': hole_id,
                    'lith_code': row['lith_code'],
                    'weathering_class': row['weathering_class'],
                    'from_m': float(row['from_m']),
                    'to_m': float(row['to_m']),
                    'mid_depth_m': mid,
                    'contact_depth_m': nearest,
                    'signed_distance_m': signed_distance,
                    'abs_distance_m': abs_distance,
                    'tgc_pct': float(row['tgc_pct']),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    for contact_bin in CONTACT_BINS:
        mask = (out['abs_distance_m'] >= contact_bin.min_dist) & (out['abs_distance_m'] < contact_bin.max_dist)
        out.loc[mask, 'distance_bin'] = contact_bin.label
        out.loc[mask, 'distance_midpoint_m'] = contact_bin.midpoint
    return out.dropna(subset=['distance_bin']).copy()


def _summarize_weathering(df: pd.DataFrame) -> pd.DataFrame:
    weathered_mask = df['lith_code'].map(_classify_weathering) == 'weathered'
    fresh = df.loc[~weathered_mask, 'tgc_pct']
    weathered = df.loc[weathered_mask, 'tgc_pct']
    fresh_mean = float(fresh.mean()) if len(fresh) else np.nan
    weathered_mean = float(weathered.mean()) if len(weathered) else np.nan
    grade_jump_pct = ((weathered_mean / fresh_mean) - 1.0) * 100.0 if fresh_mean and not np.isnan(weathered_mean) else np.nan
    return pd.DataFrame(
        [
            {
                'group': 'fresh_graphitic',
                'count': int((~weathered_mask).sum()),
                'mean_tgc_pct': fresh_mean,
                'std_tgc_pct': float(fresh.std()) if len(fresh) else np.nan,
            },
            {
                'group': 'weathered_graphitic',
                'count': int(weathered_mask.sum()),
                'mean_tgc_pct': weathered_mean,
                'std_tgc_pct': float(weathered.std()) if len(weathered) else np.nan,
            },
            {
                'group': 'weathering_upgrade',
                'count': int(weathered_mask.sum()),
                'mean_tgc_pct': grade_jump_pct,
                'std_tgc_pct': np.nan,
            },
        ]
    )


def _plot_contact_summary(summary: pd.DataFrame, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for side, sign, color in [('fresh', -1.0, '#3C6E71'), ('weathered', 1.0, '#D1495B')]:
        side_df = summary[summary['weathering_class'] == side].sort_values('distance_midpoint_m')
        if side_df.empty:
            continue
        x = sign * side_df['distance_midpoint_m'].to_numpy(dtype=float)
        y = side_df['mean_tgc_pct'].to_numpy(dtype=float)
        ax.plot(x, y, marker='o', linewidth=2, color=color, label=side.capitalize())
    ax.axvline(0.0, color='black', linewidth=1, linestyle='--')
    ax.set_xticks([-7.5, -3.5, -1.0, 1.0, 3.5, 7.5], ['5-10', '2-5', '0-2', '0-2', '2-5', '5-10'])
    ax.set_xlabel('Distance to Fresh/Weathered Contact (m)')
    ax.set_ylabel('Mean Grade (TGC %)')
    ax.set_title('Mean Grade vs Distance to Contact')
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def run(output_dir: str = 'outputs') -> dict:
    figures_dir = os.path.join(output_dir, 'figures')
    tables_dir = os.path.join(output_dir, 'tables')
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    df = _load_domain_frame(output_dir)
    weathering_summary = _summarize_weathering(df)
    contact_rows = _build_contact_rows(df)

    weathering_path = os.path.join(tables_dir, 'weathering_summary.csv')
    weathering_summary.to_csv(weathering_path, index=False)

    if contact_rows.empty:
        summary = pd.DataFrame(columns=['weathering_class', 'distance_bin', 'distance_midpoint_m', 'count', 'mean_tgc_pct', 'std_tgc_pct'])
        summary_path = os.path.join(tables_dir, 'contact_analysis.csv')
        summary.to_csv(summary_path, index=False)
        return {
            'contact_analysis': summary_path,
            'weathering_summary': weathering_path,
            'plot': None,
            'n_contacts': 0,
        }

    summary = (
        contact_rows.groupby(['weathering_class', 'distance_bin', 'distance_midpoint_m'], as_index=False)
        .agg(count=('tgc_pct', 'size'), mean_tgc_pct=('tgc_pct', 'mean'), std_tgc_pct=('tgc_pct', 'std'))
        .sort_values(['weathering_class', 'distance_midpoint_m'])
    )
    summary_path = os.path.join(tables_dir, 'contact_analysis.csv')
    summary.to_csv(summary_path, index=False)

    plot_path = os.path.join(figures_dir, 'contact_analysis.png')
    _plot_contact_summary(summary, plot_path)

    meta = {
        'n_contact_samples': int(len(contact_rows)),
        'n_holes_with_contact': int(contact_rows['hole_id'].nunique()),
        'distance_bins_m': [b.label for b in CONTACT_BINS],
    }
    with open(os.path.join(tables_dir, 'contact_analysis_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    logger.info("Saved contact analysis outputs to %s", output_dir)
    return {
        'contact_analysis': summary_path,
        'weathering_summary': weathering_path,
        'plot': plot_path,
        'n_contacts': int(len(contact_rows)),
    }


if __name__ == '__main__':
    run()
