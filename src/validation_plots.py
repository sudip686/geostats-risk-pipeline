"""
09_validation_plots.py - Validation Plots

Generates:
- Swath plots (X, Y, Z directions)
- Global histogram comparison
- QQ plot (optional)
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

def _ensure_plot_grade_column(df, grade_field='tgc_pct'):
    if 'tgc_pct' in df.columns:
        return df
    if grade_field in df.columns:
        out = df.copy()
        out['tgc_pct'] = pd.to_numeric(out[grade_field], errors='coerce')
        return out
    raise ValueError(f"Validation data missing grade column: expected 'tgc_pct' or '{grade_field}'")


def create_swath_plot(realizations, data_df, grid_def, output_path, axis=0):
    """
    Generate swath plot comparing input data to realizations.

    Args:
        realizations: 3D numpy array (n_reals, nx, ny, nz)
        data_df: DataFrame with x, y, z, grade
        grid_def: dict with x, y, z axes definitions
        output_path: Path to save
        axis: 0=X, 1=Y, 2=Z
    """
    # Define direction
    if axis == 0:
        col, grid_ax = 'x', grid_def['x']
    elif axis == 1:
        col, grid_ax = 'y', grid_def['y']
    else:
        col, grid_ax = 'z', grid_def['z']

    coords = np.asarray(grid_ax)

    # Process realizations
    dim_idx = axis
    axes_to_mean = tuple([i for i in range(1, 4) if i != (dim_idx + 1)])
    model_means = np.mean(realizations, axis=axes_to_mean)

    p10 = np.percentile(model_means, 10, axis=0)
    p50 = np.percentile(model_means, 50, axis=0)
    p90 = np.percentile(model_means, 90, axis=0)

    # Process input data
    step = grid_ax[1] - grid_ax[0] if len(grid_ax) > 1 else 1.0
    bins = np.arange(grid_ax[0], grid_ax[-1] + step, step)

    valid_data = data_df[
        (data_df[col] >= grid_ax[0]) &
        (data_df[col] <= grid_ax[-1])
    ].copy()

    valid_data['bin'] = np.digitize(valid_data[col], bins) - 1
    bin_means = valid_data.groupby('bin')['tgc_pct'].mean()

    bin_indices = np.arange(len(coords))
    data_curve = bin_means.reindex(bin_indices)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(coords, p50, 'b-', linewidth=2, label='SGS P50')
    ax.fill_between(coords, p10, p90, color='blue', alpha=0.2, label='SGS P10-P90')

    valid_mask = ~data_curve.isna()
    ax.plot(coords[valid_mask], data_curve[valid_mask], 'r--', marker='o',
            label='Input Data', markersize=4)

    ax.set_xlabel(f'{col.upper()} Coordinate (m)')
    ax.set_ylabel('Grade (TGC %)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Swath Plot - {col.upper()} Direction')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    logger.info(f"Saved swath plot: {output_path}")


def compute_swath_coverage(realizations, data_df, grid_def, axis=0):
    if axis == 0:
        col, grid_ax = 'x', grid_def['x']
    elif axis == 1:
        col, grid_ax = 'y', grid_def['y']
    else:
        col, grid_ax = 'z', grid_def['z']

    if len(grid_ax) < 2:
        return float('nan')

    p10 = np.percentile(realizations, 10, axis=0)
    p90 = np.percentile(realizations, 90, axis=0)

    step = grid_ax[1] - grid_ax[0]
    bins = np.arange(grid_ax[0], grid_ax[-1] + step, step)
    valid_data = data_df[
        (data_df[col] >= grid_ax[0]) &
        (data_df[col] <= grid_ax[-1])
    ].copy()
    if valid_data.empty:
        return float('nan')

    valid_data['bin'] = np.digitize(valid_data[col], bins) - 1
    bin_means = valid_data.groupby('bin')['tgc_pct'].mean()
    bin_indices = np.arange(len(grid_ax))
    data_curve = bin_means.reindex(bin_indices)

    valid_mask = ~data_curve.isna()
    if valid_mask.sum() == 0:
        return float('nan')

    if axis == 0:
        p10_line = np.mean(p10, axis=(1, 2))
        p90_line = np.mean(p90, axis=(1, 2))
    elif axis == 1:
        p10_line = np.mean(p10, axis=(0, 2))
        p90_line = np.mean(p90, axis=(0, 2))
    else:
        p10_line = np.mean(p10, axis=(0, 1))
        p90_line = np.mean(p90, axis=(0, 1))

    inside = (data_curve[valid_mask] >= p10_line[valid_mask]) & (data_curve[valid_mask] <= p90_line[valid_mask])
    return float(inside.sum() / valid_mask.sum()) * 100.0


def create_histogram_plot(realizations, data_df, output_path, cutoff=None, rng=None):
    """
    Compare histograms of input data vs realizations.
    """
    flat_reals = realizations.flatten()
    if rng is None:
        rng = np.random.default_rng(42)
    if len(flat_reals) > 1000000:
        flat_reals = rng.choice(flat_reals, 1000000, replace=False)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(flat_reals, bins=50, density=True, alpha=0.5,
            color='blue', label='SGS Realizations')
    ax.hist(data_df['tgc_pct'], bins=20, density=True, alpha=0.5,
            color='red', label='Input Data')

    ax.set_xlabel('Grade (TGC %)')
    ax.set_ylabel('Density')
    ax.legend()
    title = 'Global Histogram Validation'
    if cutoff is not None:
        title += f' (cutoff {cutoff:.1f}% TGC)'
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    logger.info(f"Saved histogram: {output_path}")


def create_qq_plot(data_values, simulated_values, output_path, n_points=1000, cutoff=None):
    """
    Create Q-Q plot comparing data quantiles to simulation quantiles.
    """
    # Sample if too large
    rng = np.random.default_rng(42)
    if len(simulated_values) > n_points:
        idx = rng.choice(len(simulated_values), n_points, replace=False)
        sim_sample = simulated_values[idx]
    else:
        sim_sample = simulated_values

    # Compute quantiles
    n = len(sim_sample)
    data_quantiles = np.percentile(data_values, np.linspace(0, 100, n))
    sim_quantiles = np.percentile(sim_sample, np.linspace(0, 100, n))

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(data_quantiles, sim_quantiles, alpha=0.5, s=20)

    # Add 1:1 line
    min_val = min(data_quantiles.min(), sim_quantiles.min())
    max_val = max(data_quantiles.max(), sim_quantiles.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='1:1 Line')

    ax.set_xlabel('Input Data Quantiles')
    ax.set_ylabel('Simulated Quantiles')
    title = 'Q-Q Plot: Data vs SGS'
    if cutoff is not None:
        title += f' (cutoff {cutoff:.1f}% TGC)'
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    logger.info(f"Saved QQ plot: {output_path}")


def compute_qq_rmse(data_values, simulated_values, n_points=1000):
    rng = np.random.default_rng(42)
    if len(simulated_values) > n_points:
        idx = rng.choice(len(simulated_values), n_points, replace=False)
        sim_sample = simulated_values[idx]
    else:
        sim_sample = simulated_values

    n = len(sim_sample)
    data_quantiles = np.percentile(data_values, np.linspace(0, 100, n))
    sim_quantiles = np.percentile(sim_sample, np.linspace(0, 100, n))
    rmse = np.sqrt(np.mean((data_quantiles - sim_quantiles) ** 2))
    return float(rmse)


def compute_histogram_overlap(data_values, sim_values, bins=50):
    """Compute histogram overlap between data and simulations."""
    data_values = np.asarray(data_values)
    sim_values = np.asarray(sim_values)

    min_val = min(data_values.min(), sim_values.min())
    max_val = max(data_values.max(), sim_values.max())

    hist_bins = np.linspace(min_val, max_val, bins + 1)
    data_hist, _ = np.histogram(data_values, bins=hist_bins, density=True)
    sim_hist, _ = np.histogram(sim_values, bins=hist_bins, density=True)

    bin_width = hist_bins[1] - hist_bins[0]
    overlap = np.sum(np.minimum(data_hist, sim_hist)) * bin_width
    return float(overlap)


def compute_swath_corr(realizations, data_df, grid_def, axis=0):
    """Compute correlation between data and SGS P50 swath in a direction."""
    if axis == 0:
        col, grid_ax = 'x', grid_def['x']
    elif axis == 1:
        col, grid_ax = 'y', grid_def['y']
    else:
        col, grid_ax = 'z', grid_def['z']

    if len(grid_ax) < 2:
        return float('nan')

    dim_idx = axis
    axes_to_mean = tuple([i for i in range(1, 4) if i != (dim_idx + 1)])
    model_means = np.mean(realizations, axis=axes_to_mean)
    p50 = np.percentile(model_means, 50, axis=0)

    step = grid_ax[1] - grid_ax[0]
    bins = np.arange(grid_ax[0], grid_ax[-1] + step, step)
    valid_data = data_df[
        (data_df[col] >= grid_ax[0]) &
        (data_df[col] <= grid_ax[-1])
    ].copy()
    if valid_data.empty:
        return float('nan')

    valid_data['bin'] = np.digitize(valid_data[col], bins) - 1
    bin_means = valid_data.groupby('bin')['tgc_pct'].mean()
    bin_indices = np.arange(len(grid_ax))
    data_curve = bin_means.reindex(bin_indices)

    valid_mask = ~data_curve.isna()
    if valid_mask.sum() < 2:
        return float('nan')

    return float(np.corrcoef(data_curve[valid_mask], p50[valid_mask])[0, 1])


def load_validation_data(data_path, data_dir, config):
    """Load validation data, preferring external reference if configured."""
    validation_cfg = config.get('validation', {}) if config else {}
    ref_path = validation_cfg.get('reference_data')
    if config and config.get('calibration', {}).get('enabled'):
        ref_path = config.get('calibration', {}).get('reference_data', ref_path)

    if ref_path:
        df = pd.read_csv(ref_path, low_memory=False)
        col_map = validation_cfg.get('reference_columns', {})
        if config and config.get('calibration', {}).get('enabled'):
            col_map = config.get('calibration', {}).get('reference_columns', col_map)
        x_col = col_map.get('x', 'X')
        y_col = col_map.get('y', 'Y')
        z_col = col_map.get('z', 'Z')
        grade_col = col_map.get('grade', 'TGC_%')

        for col in [x_col, y_col, z_col, grade_col]:
            if col not in df.columns:
                raise ValueError(f"Validation data missing column: {col}")

        df = df[[x_col, y_col, z_col, grade_col]].copy()
        df.columns = ['x', 'y', 'z', 'tgc_pct']
        df['x'] = pd.to_numeric(df['x'], errors='coerce')
        df['y'] = pd.to_numeric(df['y'], errors='coerce')
        df['z'] = pd.to_numeric(df['z'], errors='coerce')
        df['tgc_pct'] = pd.to_numeric(df['tgc_pct'], errors='coerce')
        df = df.dropna(subset=['x', 'y', 'z', 'tgc_pct'])
        logger.info(f"Using validation reference data: {ref_path} ({len(df)} rows)")
        return df

    if data_path:
        return pd.read_csv(data_path)

    from .domains import run as run_domains
    data_df, _ = run_domains(data_dir=data_dir)
    return data_df


def load_swath_data(data_dir='data', config=None):
    """Load swath validation data from composites/domains."""
    from .domains import run as run_domains
    grade_field = config.get('grade_field', 'tgc_pct') if config else 'tgc_pct'
    data_df, _ = run_domains(
        data_dir=data_dir,
        target_lith_codes=config.get('target_lith_codes') if config else None,
        grade_field=grade_field,
    )
    return data_df


def _grid_def_from_meta(meta_path):
    with open(meta_path, 'r', encoding='utf-8') as f:
        grid_meta = json.load(f)
    return {
        'x': grid_meta['x_min'] + np.arange(int(grid_meta['nx'])) * float(grid_meta['dx']),
        'y': grid_meta['y_min'] + np.arange(int(grid_meta['ny'])) * float(grid_meta['dy']),
        'z': grid_meta['z_min'] + np.arange(int(grid_meta['nz'])) * float(grid_meta['dz']),
        'dx': float(grid_meta['dx']),
        'dy': float(grid_meta['dy']),
        'dz': float(grid_meta['dz']),
    }


def _resolve_validation_support(grids_dir, config=None, mode='auto'):
    reporting_reals = os.path.join(grids_dir, 'sgs_reals_reporting.npy')
    reporting_meta = os.path.join(grids_dir, 'sgs_reporting_meta.json')
    sim_reals = os.path.join(grids_dir, 'sgs_reals.npy')
    sim_meta = os.path.join(grids_dir, 'sgs_meta.json')

    if mode == 'reporting_support':
        if not (os.path.exists(reporting_reals) and os.path.exists(reporting_meta)):
            raise FileNotFoundError("Reporting-support validation requested but reporting outputs are missing")
        return reporting_reals, reporting_meta, 'reporting_support'

    if mode == 'simulation_support':
        if not (os.path.exists(sim_reals) and os.path.exists(sim_meta)):
            raise FileNotFoundError("Simulation-support validation requested but simulation outputs are missing")
        return sim_reals, sim_meta, 'simulation_support'

    if config and config.get('reporting_grid') and os.path.exists(reporting_reals) and os.path.exists(reporting_meta):
        return reporting_reals, reporting_meta, 'reporting_support'
    if os.path.exists(sim_reals) and os.path.exists(sim_meta):
        return sim_reals, sim_meta, 'simulation_support'
    raise FileNotFoundError("Unable to resolve validation support files")


def run(
    realizations_path=None,
    data_path=None,
    data_dir='data',
    output_dir='outputs/figures',
    cutoff=3.0,
    config=None,
    mode='auto',
    suffix='',
    metrics_filename='validation_metrics.json',
):
    """
    Run validation plotting.

    Args:
        realizations_path: Path to sgs_reals.npy
        data_path: Path to domain CSV
        data_dir: Input data directory
        output_dir: Output directory

    Returns:
        dict: Paths to generated plots
    """
    outputs_root = output_dir
    if os.path.basename(output_dir) == 'figures':
        outputs_root = os.path.dirname(output_dir)
    grids_dir = os.path.join(outputs_root, 'grids')
    figures_dir = os.path.join(outputs_root, 'figures')
    tables_dir = os.path.join(outputs_root, 'tables')
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    # Load realizations
    if realizations_path:
        realizations = np.load(realizations_path)
        grids_dir = os.path.dirname(realizations_path)
        grid_meta_path = os.path.join(grids_dir, 'sgs_meta.json')
        support_mode = mode if mode != 'auto' else 'simulation_support'
    else:
        selected_reals, grid_meta_path, support_mode = _resolve_validation_support(grids_dir, config=config, mode=mode)
        if support_mode == 'simulation_support' and config and config.get('calibration', {}).get('enabled'):
            calibrated_path = os.path.join(grids_dir, 'sgs_reals_calibrated.npy')
            realizations = np.load(calibrated_path if os.path.exists(calibrated_path) else selected_reals)
        else:
            realizations = np.load(selected_reals)

    # Load data
    grade_field = config.get('grade_field', 'tgc_pct') if config else 'tgc_pct'
    data_df = load_validation_data(data_path, data_dir, config)
    swath_df = load_swath_data(data_dir=data_dir, config=config)
    data_df = _ensure_plot_grade_column(data_df, grade_field=grade_field)
    swath_df = _ensure_plot_grade_column(swath_df, grade_field=grade_field)

    if os.path.exists(grid_meta_path):
        grid_def = _grid_def_from_meta(grid_meta_path)
    else:
        # Default grid
        grid_def = {
            'x': np.arange(0, 1000, 25),
            'y': np.arange(0, 1000, 25),
            'z': np.arange(0, 100, 5),
            'dx': 25,
            'dy': 25,
            'dz': 5,
        }

    rng = np.random.default_rng(42)

    # Generate plots
    create_swath_plot(realizations, swath_df, grid_def,
                     os.path.join(figures_dir, f'swath_x{suffix}.png'), axis=0)
    create_swath_plot(realizations, swath_df, grid_def,
                     os.path.join(figures_dir, f'swath_y{suffix}.png'), axis=1)
    create_swath_plot(realizations, swath_df, grid_def,
                     os.path.join(figures_dir, f'swath_z{suffix}.png'), axis=2)
    create_histogram_plot(realizations, data_df,
                         os.path.join(figures_dir, f'histogram_validation{suffix}.png'), cutoff=cutoff, rng=rng)

    # QQ plot
    flat_sim = realizations.flatten()
    if len(flat_sim) > 10000:
        flat_sim = rng.choice(flat_sim, 10000, replace=False)
    create_qq_plot(data_df['tgc_pct'].values, flat_sim,
                  os.path.join(figures_dir, f'qq_plot{suffix}.png'), cutoff=cutoff)

    # Validation metrics
    mean_data = float(data_df['tgc_pct'].mean())
    std_data = float(data_df['tgc_pct'].std())
    mean_sim = float(np.mean(flat_sim))
    std_sim = float(np.std(flat_sim))
    hist_overlap = compute_histogram_overlap(data_df['tgc_pct'].values, flat_sim)

    metrics = {
        'mean_data': mean_data,
        'std_data': std_data,
        'mean_sim': mean_sim,
        'std_sim': std_sim,
        'hist_overlap': hist_overlap,
        'qq_rmse': compute_qq_rmse(data_df['tgc_pct'].values, flat_sim),
        'swath_corr_x': compute_swath_corr(realizations, swath_df, grid_def, axis=0),
        'swath_corr_y': compute_swath_corr(realizations, swath_df, grid_def, axis=1),
        'swath_corr_z': compute_swath_corr(realizations, swath_df, grid_def, axis=2),
        'swath_coverage_pct': np.nanmean([
            compute_swath_coverage(realizations, swath_df, grid_def, axis=0),
            compute_swath_coverage(realizations, swath_df, grid_def, axis=1),
            compute_swath_coverage(realizations, swath_df, grid_def, axis=2),
        ]),
        'validation_mode': support_mode,
        'support_dx': float(grid_def['dx']),
        'support_dy': float(grid_def['dy']),
        'support_dz': float(grid_def['dz']),
    }

    metrics_path = os.path.join(tables_dir, metrics_filename)
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    # Regression check: recompute key metrics from arrays used in plots
    recomputed = {
        'mean_data': float(data_df['tgc_pct'].mean()),
        'std_data': float(data_df['tgc_pct'].std()),
        'mean_sim': float(np.mean(flat_sim)),
        'std_sim': float(np.std(flat_sim)),
        'hist_overlap': compute_histogram_overlap(data_df['tgc_pct'].values, flat_sim),
    }
    for key, val in recomputed.items():
        if not np.isclose(metrics[key], val, rtol=1e-6, atol=1e-6):
            raise ValueError(f"Validation metric mismatch for {key}")

    logger.info(f"Generated validation plots in {output_dir}")

    return {
        'mode': support_mode,
        'metrics': metrics_path,
        'swath_x': os.path.join(figures_dir, f'swath_x{suffix}.png'),
        'swath_y': os.path.join(figures_dir, f'swath_y{suffix}.png'),
        'swath_z': os.path.join(figures_dir, f'swath_z{suffix}.png'),
        'histogram': os.path.join(figures_dir, f'histogram_validation{suffix}.png'),
        'qq': os.path.join(figures_dir, f'qq_plot{suffix}.png')
    }
