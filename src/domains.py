"""
03_domains.py - Domain Analysis

Analyzes and filters data by lithology domains.
Computes domain statistics (n, mean, std, CV).
"""

import logging

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .variography import build_orebody_axes, orebody_from_config

logger = logging.getLogger(__name__)


DEFAULT_CANONICAL_DOMAIN_GROUPS = {
    "fresh_graphitic": ["GRSC", "GRSC1", "GRSC2"],
    "weathered_graphitic": ["SAP (GRSC)", "SAPR (GRSC)"],
    "host_waste": [],
}

LITH_CODE_ALIASES = {
    "SAP": "SAP (GRSC)",
    "SAPR": "SAPR (GRSC)",
    "SAP (GRSC2)": "SAP (GRSC)",
    "SAPR (GRSC2)": "SAPR (GRSC)",
    "GRSC-3": "GRSC3",
}


def canonical_lith_code(lith_code):
    """Return a canonical lithology token for matching config domain groups."""
    lith = str(lith_code).strip()
    return LITH_CODE_ALIASES.get(lith.upper(), lith)


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
        target = {canonical_lith_code(code) for code in target_lith_codes}
        domain_data = composites[composites['lith_code'].map(canonical_lith_code).isin(target)]
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

    target = {canonical_lith_code(code) for code in domain_codes}
    filtered = composites[composites['lith_code'].map(canonical_lith_code).isin(target)].copy()
    filtered = filtered[filtered[grade_field].notna()]

    logger.info(f"Domain filter: {len(composites)} -> {len(filtered)} samples")
    return filtered


def canonical_domain_groups(config=None):
    """Return canonical hard-boundary domain groups from config."""
    groups_cfg = ((config or {}).get('domains') or {}).get('canonical_groups') or {}
    groups = {}
    for name, payload in groups_cfg.items():
        if isinstance(payload, dict):
            lith_codes = payload.get('lith_codes') or []
            group_payload = dict(payload)
        else:
            lith_codes = payload or []
            group_payload = {}
        lith_codes = [str(code) for code in lith_codes]
        if lith_codes or str(name) == 'host_waste':
            groups[str(name)] = {
                'name': str(name),
                'lith_codes': lith_codes,
                **group_payload,
            }
    if groups:
        return groups
    return {
        name: {'name': name, 'lith_codes': list(codes)}
        for name, codes in DEFAULT_CANONICAL_DOMAIN_GROUPS.items()
    }


def lith_code_to_domain_group(lith_code, config=None):
    """Map a lithology code to a canonical categorical domain."""
    lith = canonical_lith_code(lith_code)
    groups = canonical_domain_groups(config=config)
    for name, payload in groups.items():
        if name == 'host_waste':
            continue
        targets = {canonical_lith_code(code) for code in payload.get('lith_codes', [])}
        if lith in targets:
            return name
    lith_upper = lith.upper().replace(" ", "")
    if lith_upper.startswith("SAP") and "GRSC" in lith_upper:
        return "weathered_graphitic"
    if "GRSC" in lith_upper:
        return "fresh_graphitic"
    return 'host_waste'


def build_categorical_domain_data(composites, config=None, grade_field='tgc_pct'):
    """Return full composite table annotated with categorical domain labels."""
    result = composites.copy()
    result = result[result[grade_field].notna()].copy()
    result['domain_group'] = result['lith_code'].map(lambda code: lith_code_to_domain_group(code, config=config))
    return result


def split_domain_groups(composites, config=None, grade_field='tgc_pct'):
    """Split composites into canonical geology-led domain groups."""
    groups = canonical_domain_groups(config=config)
    composites = build_categorical_domain_data(composites, config=config, grade_field=grade_field)
    out = {}
    for name, payload in groups.items():
        subset = composites[composites['domain_group'] == name].copy()
        subset = subset[subset[grade_field].notna()]
        out[name] = {
            **payload,
            'data': subset,
            'stats': analyze_domains(subset, None, grade_field=grade_field),
        }
    return out


def _project_coords(coords: np.ndarray, config=None) -> np.ndarray:
    orebody = orebody_from_config(config)
    if not orebody:
        return coords.astype(float)

    strike_deg = orebody.get('strike_deg')
    dip_deg = orebody.get('dip_deg')
    dip_direction_deg = orebody.get('dip_direction_deg')
    dip_positive_down = bool(orebody.get('dip_positive_down', True))
    if strike_deg is None or dip_deg is None:
        return coords.astype(float)

    axes = build_orebody_axes(
        float(strike_deg),
        float(dip_deg),
        float(dip_direction_deg) if dip_direction_deg is not None else None,
        dip_positive_down=dip_positive_down,
    )
    basis = np.vstack([axes['strike'], axes['dip'], axes['normal']])
    return coords @ basis.T


def assign_domain_masks(grid_def, composites, config=None):
    """Assign each simulation cell to the nearest canonical domain composite."""
    grouped = split_domain_groups(composites, config=config)
    if not grouped:
        raise ValueError("No canonical domain groups could be constructed from the composites")

    labels = []
    sample_points = []
    for idx, (name, payload) in enumerate(grouped.items()):
        subset = payload['data'][['x', 'y', 'z']].dropna().to_numpy(dtype=float)
        if subset.size == 0:
            raise ValueError(f"Domain group '{name}' has no valid XYZ composites for mask assignment")
        labels.append(np.full(subset.shape[0], idx, dtype=int))
        sample_points.append(subset)

    sample_points = np.vstack(sample_points)
    sample_labels = np.concatenate(labels)
    projected_samples = _project_coords(sample_points, config=config)
    tree = cKDTree(projected_samples)

    grid_points = np.array(np.meshgrid(grid_def['x'], grid_def['y'], grid_def['z'], indexing='ij')).reshape(3, -1).T
    projected_grid = _project_coords(grid_points, config=config)
    _, nearest_idx = tree.query(projected_grid, k=1)
    assigned = sample_labels[np.asarray(nearest_idx, dtype=int)].reshape(
        len(grid_def['x']),
        len(grid_def['y']),
        len(grid_def['z']),
    )

    masks = {}
    summary = {}
    group_names = list(grouped.keys())
    total = int(assigned.size)
    for idx, name in enumerate(group_names):
        mask = assigned == idx
        masks[name] = mask
        summary[name] = {
            'n_cells': int(mask.sum()),
            'cell_fraction': float(mask.sum() / max(1, total)),
            'lith_codes': grouped[name]['lith_codes'],
        }

    return masks, {
        'assignment_method': str((((config or {}).get('domains') or {}).get('assignment') or {}).get('method', 'nearest_composite')),
        'group_order': group_names,
        'summary': summary,
    }


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
