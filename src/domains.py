"""
03_domains.py - Domain Analysis

Analyzes and filters data by lithology domains.
Computes domain statistics (n, mean, std, CV).
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def analyze_domains(composites, target_lith_codes=None, grade_field='tgc_pct'):
    """
    Analyze lithology domains and compute statistics.

    Args:
        composites (pd.DataFrame): Composited data with lith_code and grade
        target_lith_codes: List of lithology codes to include (None = all)
        grade_field: Grade column name

    Returns:
        dict: Domain statistics by lith_code
    """
    stats = {}

    if target_lith_codes:
        domain_data = composites[composites['lith_code'].isin(target_lith_codes)]
    else:
        domain_data = composites

    for lith_code in domain_data['lith_code'].unique():
        data = domain_data[domain_data['lith_code'] == lith_code][grade_field]
        data = data[~np.isnan(data)]

        if len(data) > 0:
            stats[lith_code] = {
                'n': len(data),
                'mean': float(data.mean()),
                'std': float(data.std()),
                'cv': float(data.std() / data.mean()) if data.mean() > 0 else 0,
                'min': float(data.min()),
                'max': float(data.max()),
                'median': float(data.median())
            }

    # Log summary
    logger.info("Domain Statistics:")
    for code, s in stats.items():
        logger.info(f"  {code}: n={s['n']}, mean={s['mean']:.2f}%, std={s['std']:.2f}, CV={s['cv']:.2f}")

    return stats


def filter_domain(composites, domain_codes, grade_field='tgc_pct'):
    """
    Filter composites by domain codes.

    Args:
        composites (pd.DataFrame): Composited data
        domain_codes: Single code or list of codes
        grade_field: Grade column name

    Returns:
        pd.DataFrame: Filtered data
    """
    if isinstance(domain_codes, str):
        domain_codes = [domain_codes]

    filtered = composites[composites['lith_code'].isin(domain_codes)].copy()
    filtered = filtered[filtered[grade_field].notna()]

    logger.info(f"Domain filter: {len(composites)} -> {len(filtered)} samples")
    return filtered


def run(composites_path=None, data_dir='data', target_lith_codes=None, grade_field='tgc_pct', output_path=None):
    """
    Run domain analysis.

    Args:
        composites_path: Path to composites CSV (if None, runs compositing first)
        data_dir: Input data directory
        target_lith_codes: List of lith codes to include
        grade_field: Grade column name
        output_path: Optional output path

    Returns:
        tuple: (filtered_df, stats_dict)
    """
    if composites_path:
        composites = pd.read_csv(composites_path)
    else:
        from .composite import run as run_composite
        composites = run_composite(data_dir=data_dir, grade_field=grade_field)

    # Analyze domains
    stats = analyze_domains(composites, target_lith_codes, grade_field)

    # Filter
    if target_lith_codes:
        filtered = filter_domain(composites, target_lith_codes, grade_field)
    else:
        filtered = composites

    if output_path:
        filtered.to_csv(output_path, index=False)
        logger.info(f"Saved domain data to {output_path}")

    return filtered, stats
