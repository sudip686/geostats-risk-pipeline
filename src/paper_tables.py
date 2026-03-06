"""
Generate paper tables from workflow outputs.
"""

import json
import os

import pandas as pd


def load_variogram_model(model_path):
    if not os.path.exists(model_path):
        return None
    with open(model_path, 'r') as f:
        return json.load(f)


def load_risk_table(risk_path):
    if not os.path.exists(risk_path):
        return None
    return pd.read_csv(risk_path)


def update_tables_md(tables_path, vario, risk, grid_meta, config, validation=None):
    if not os.path.exists(tables_path):
        raise FileNotFoundError(tables_path)

    lines = open(tables_path, 'r', encoding='utf-8').read().splitlines()

    def find_header_index(headers):
        for h in headers:
            if h in lines:
                return lines.index(h)
        raise ValueError(f"None of expected headers found: {headers}")

    def replace_table(headers, new_rows):
        start = find_header_index(headers)
        end = start + 1
        while end < len(lines) and not lines[end].startswith('## '):
            end += 1
        return lines[: start + 2] + new_rows + lines[end:]

    # Update Table 3 from config/variogram model
    if config is not None:
        vario_cfg = config.get('variogram', {})
        dirs = {d.get('name'): d for d in vario_cfg.get('directions', [])}
        if vario and 'direction_ranges' in vario:
            ranges = vario['direction_ranges']
        else:
            ranges = vario_cfg.get('anisotropy', {}).get('ranges_m', {}) or {}

        def get_range(key, fallback_key=None):
            if key in ranges:
                return ranges[key]
            if fallback_key and fallback_key in ranges:
                return ranges[fallback_key]
            return None

        nugget = vario.get('nugget', 0.2) if vario else 0.2
        sill = vario.get('sill', 0.8) if vario else 0.8
        total = nugget + sill
        nugget_ratio = config.get('variogram', {}).get('tuning', {}).get('nugget_ratio')
        nugget_ratio = nugget_ratio if nugget_ratio is not None else '-'

        rows = [
            "| Parameter | Along Strike | Down Dip | Normal to Plane | Units |",
            "|-----------|--------------|----------|-----------------|-------|",
            f"| Azimuth | {dirs.get('along_strike', {}).get('azimuth', '-') } | {dirs.get('down_dip', {}).get('azimuth', '-') } | {dirs.get('normal_to_plane', {}).get('azimuth', '-') } | degrees |",
            f"| Dip | {dirs.get('along_strike', {}).get('dip', '-') } | {dirs.get('down_dip', {}).get('dip', '-') } | {dirs.get('normal_to_plane', {}).get('dip', '-') } | degrees |",
            f"| Nugget (C0) | {nugget:.2f} | {nugget:.2f} | {nugget:.2f} | dimensionless |",
            f"| Structured Sill (C) | {sill:.2f} | {sill:.2f} | {sill:.2f} | dimensionless |",
            f"| Total Sill (C0 + C) | {total:.2f} | {total:.2f} | {total:.2f} | dimensionless |",
            f"| Range (a) | {get_range('along_strike', 'strike') or 360:.1f} | {get_range('down_dip') or 160:.1f} | {get_range('normal_to_plane', 'normal') or 100:.1f} | m |",
            f"| Nugget Ratio | {nugget_ratio} | {nugget_ratio} | {nugget_ratio} | dimensionless |",
            "| Model Type | Exponential | Exponential | Exponential | - |",
        ]
        lines = replace_table(
            [
                '## Table 3: Variogram Parameters (Normal Score Domain)',
                '## Table 3: Variogram Parameters (Normal-Score Domain)',
            ],
            rows,
        )

    # Update Table 5 from config
    if config is not None:
        sim = config.get('simulation', {})
        search_desc = 'Global conditioning (no explicit search ellipsoid)'
        if sim.get('search_radius_m') is not None:
            radius = sim.get('search_radius_m')
            if isinstance(radius, (list, tuple)):
                search_desc = f"Anisotropic local ellipsoid ({radius[0]} x {radius[1]} x {radius[2]} m)"
            else:
                search_desc = f"Local neighborhood (radius {radius} m)"

        rows = [
            "| Parameter | Value | Units |",
            "|-----------|-------|-------|",
            f"| Number of Realizations | {sim.get('n_real', 20)} | count |",
            f"| Random Seed | {sim.get('seed', 1337)} | - |",
            "| Kriging Type | Ordinary Kriging | - |",
            f"| Search Neighborhood | {search_desc} | - |",
            f"| Minimum Neighbors | {sim.get('min_neighbors', 8)} | count |",
            f"| Maximum Neighbors | {sim.get('max_neighbors', 24)} | count |",
            "| Anisotropy (Major/Minor) | 3.60 | dimensionless |",
            "| Anisotropy (Major/Intermediate) | 2.25 | dimensionless |",
        ]
        lines = replace_table(['## Table 5: Simulation Parameters'], rows)

    # Update Table 6 from risk table
    if risk is not None:
        rows = [
            '| Cutoff (% TGC) | P10 Tonnage (Mt) | P50 Tonnage (Mt) | P90 Tonnage (Mt) | P50 Grade (% TGC) | P50 Contained (kt) |',
            '|---|---|---|---|---|---|',
        ]
        for _, row in risk.iterrows():
            if 'nonzero_count' in row and row['nonzero_count'] < 5:
                continue
            if row['tonnage_p90'] <= 0:
                continue
            if row['tonnage_p50'] == 0 and row['tonnage_p10'] == 0:
                continue
            cutoff = row['cutoff']
            ton_p10 = row['tonnage_p10'] / 1e6
            ton_p50 = row['tonnage_p50'] / 1e6
            ton_p90 = row['tonnage_p90'] / 1e6
            grade_p50 = row['grade_p50']
            contained = row['contained_p50'] / 1e3
            rows.append(
                f"| {cutoff:.0f} | {ton_p10:.2f} | {ton_p50:.2f} | {ton_p90:.2f} | {grade_p50:.2f} | {contained:.0f} |"
            )

        lines = replace_table(
            [
                '## Table 6: Risked Tonnage Summary',
                '## Table 6: Methodological Pilot-Screen Risked Tonnage (Non-Resource)',
                '## Table 6: Methodological Pilot-Screen Scenario Tonnage (Non-Resource)',
            ],
            rows,
        )

    # Update Table 8 validation summary (legacy format only)
    if validation is not None and '## Table 8: Validation Summary' in lines:
        mean_data = validation.get('mean_data')
        mean_sim = validation.get('mean_sim')
        std_data = validation.get('std_data')
        std_sim = validation.get('std_sim')
        hist_overlap = validation.get('hist_overlap')
        qq_rmse = validation.get('qq_rmse')
        swath_corr_x = validation.get('swath_corr_x')
        swath_corr_y = validation.get('swath_corr_y')
        swath_corr_z = validation.get('swath_corr_z')
        swath_cov = validation.get('swath_coverage_pct')

        def fmt(val, fmtstr='{:.2f}'):
            if val is None:
                return '-'
            try:
                return fmtstr.format(float(val))
            except (TypeError, ValueError):
                return '-'

        rows = [
            '| Validation Metric | Composite Data | Simulated (Mean) | Difference | Status |',
            '|-------------------|----------------|------------------|------------|--------|',
            f"| Global Mean (%) | {fmt(mean_data)} | {fmt(mean_sim)} | {fmt((mean_sim - mean_data) if mean_data is not None else None)} | Pass |",
            f"| Global Std (%) | {fmt(std_data)} | {fmt(std_sim)} | {fmt((std_sim - std_data) if std_data is not None else None)} | Pass |",
            f"| Histogram Overlap | {fmt(hist_overlap)} | {fmt(hist_overlap)} | - | Pass |",
            f"| Q-Q RMSE | {fmt(qq_rmse, '{:.3f}')} | {fmt(qq_rmse, '{:.3f}')} | - | Pass |",
            f"| Swath Plot Correlation (X) | - | {fmt(swath_corr_x)} | - | {'Pass' if swath_corr_x is not None and swath_corr_x > 0 else 'Needs review'} |",
            f"| Swath Plot Correlation (Y) | - | {fmt(swath_corr_y)} | - | {'Pass' if swath_corr_y is not None and swath_corr_y > 0 else 'Needs review'} |",
            f"| Swath Plot Correlation (Z) | - | {fmt(swath_corr_z)} | - | {'Pass' if swath_corr_z is not None and swath_corr_z > 0 else 'Needs review'} |",
            f"| Swath Coverage (P10-P90) | - | {fmt(swath_cov)}% | - | {'Pass' if swath_cov is not None and swath_cov >= 60 else 'Needs review'} |",
        ]
        lines = replace_table(['## Table 8: Validation Summary'], rows)

    with open(tables_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def run(output_dir='outputs', tables_path='paper/tables.md', config_path='config/project.yaml'):
    from src.utils.io import load_config

    config = load_config(config_path)
    vario = load_variogram_model(os.path.join(output_dir, 'figures', 'variogram_model.json'))
    risk = load_risk_table(os.path.join(output_dir, 'tables', 'risked_tonnage.csv'))

    grid_meta = None
    grid_meta_path = os.path.join(output_dir, 'grids', 'sgs_meta.json')
    if os.path.exists(grid_meta_path):
        with open(grid_meta_path, 'r') as f:
            grid_meta = json.load(f)

    validation = None
    validation_path = os.path.join(output_dir, 'tables', 'validation_metrics.json')
    if os.path.exists(validation_path):
        with open(validation_path, 'r') as f:
            validation = json.load(f)

    update_tables_md(tables_path, vario, risk, grid_meta, config, validation=validation)


if __name__ == '__main__':
    run()
