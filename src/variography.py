"""
variography.py - Variography

Computes directional variograms and fits anisotropic models.
Supports omnidirectional and directional variograms with anisotropy.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import csv

logger = logging.getLogger(__name__)

# Try to import gstools
try:
    import gstools as gs
    GSTOOLS_AVAILABLE = True
except ImportError:
    GSTOOLS_AVAILABLE = False
    logger.warning("gstools not available. Variography will be limited.")


def estimate_directional_variogram(
    coords,
    values,
    azimuth,
    dip,
    tolerance=22.5,
    n_lags=10,
    max_dist=200,
    max_pairs=200000,
    dip_positive_down=True,
    return_debug=False,
):
    """
    Estimate directional variogram along a specific direction.

    Args:
        coords: tuple of (x, y, z) arrays
        values: array of values
        azimuth: azimuth angle in degrees (0-360, clockwise from North)
        dip: dip angle in degrees (-90 to 90, positive is down)
        tolerance: angular tolerance in degrees
        n_lags: number of lags
        max_dist: maximum distance

    Returns:
        bin_centers, gamma values, counts
        if return_debug=True, also returns debug dict
    """
    x, y, z = coords

    # Compute direction vector (dip-positive-down)
    dir_vec = az_dip_to_unit_vector(azimuth, dip, dip_positive_down=dip_positive_down)
    dir_x, dir_y, dir_z = dir_vec

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    values = np.asarray(values, dtype=float)
    n = len(x)
    angle_bins = np.arange(0.0, 95.0, 5.0)
    angle_hist = np.zeros(len(angle_bins) - 1, dtype=float)
    debug = {
        'n_points': int(n),
        'possible_pairs': int(n * (n - 1) // 2),
        'target_pairs': 0,
        'attempts': 0,
        'non_self_pairs': 0,
        'distance_filtered_pairs': 0,
        'angle_evaluated_pairs': 0,
        'angle_accepted_pairs': 0,
        'binned_pairs': 0,
        'nonzero_lags': 0,
        'total_lag_pairs': 0,
        'angle_hist_bins_deg': angle_bins.tolist(),
        'angle_hist_counts': angle_hist.tolist(),
    }
    if n < 2:
        if return_debug:
            return np.array([]), np.array([]), np.array([]), debug
        return np.array([]), np.array([]), np.array([])

    # Bin by true separation distance. Directionality is controlled by angular tolerance.
    bins = np.linspace(0, max_dist, n_lags + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    counts = np.zeros(n_lags, dtype=float)
    sum_sq = np.zeros(n_lags, dtype=float)

    rng = np.random.default_rng(1337)
    total_pairs = n * (n - 1) // 2
    target_pairs = min(total_pairs, max_pairs)
    debug['target_pairs'] = int(target_pairs)

    accepted = 0
    attempts = 0
    max_attempts = 25
    batch = min(max(20000, max_pairs), 200000)

    while accepted < target_pairs and attempts < max_attempts:
        attempts += 1
        i = rng.integers(0, n, size=batch)
        j = rng.integers(0, n, size=batch)
        keep = i != j
        if not np.any(keep):
            continue

        i = i[keep]
        j = j[keep]
        debug['non_self_pairs'] += int(len(i))

        # Canonicalize pair ordering (undirected pairs).
        lo = np.minimum(i, j)
        hi = np.maximum(i, j)
        i = lo
        j = hi

        dx = x[j] - x[i]
        dy = y[j] - y[i]
        dz_val = z[j] - z[i]
        h = np.sqrt(dx * dx + dy * dy + dz_val * dz_val)

        dist_ok = (h >= 1.0) & (h <= max_dist)
        if not np.any(dist_ok):
            continue

        debug['distance_filtered_pairs'] += int(np.count_nonzero(dist_ok))
        dx = dx[dist_ok]
        dy = dy[dist_ok]
        dz_val = dz_val[dist_ok]
        h = h[dist_ok]
        i = i[dist_ok]
        j = j[dist_ok]

        proj = dx * dir_x + dy * dir_y + dz_val * dir_z
        cos_angle = np.clip(np.abs(proj) / h, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        hist_counts, _ = np.histogram(angle, bins=angle_bins)
        angle_hist += hist_counts
        debug['angle_evaluated_pairs'] += int(len(angle))
        ang_ok = angle <= tolerance
        if not np.any(ang_ok):
            continue

        debug['angle_accepted_pairs'] += int(np.count_nonzero(ang_ok))
        h_dir = h[ang_ok]
        i_dir = i[ang_ok]
        j_dir = j[ang_ok]

        gamma_sq = (values[i_dir] - values[j_dir]) ** 2
        # Bins are [left, right), final bin includes the edge.
        bin_idx = np.searchsorted(bins, h_dir, side="right") - 1
        valid_bin = (bin_idx >= 0) & (bin_idx < n_lags)
        if not np.any(valid_bin):
            continue

        bin_idx = bin_idx[valid_bin]
        gamma_sq = gamma_sq[valid_bin]
        counts += np.bincount(bin_idx, minlength=n_lags)
        sum_sq += np.bincount(bin_idx, weights=gamma_sq, minlength=n_lags)
        accepted += int(valid_bin.sum())

    if counts.sum() < 10:
        logger.warning(f"Too few pairs for direction {azimuth}/{dip}: {int(counts.sum())}")
        debug['attempts'] = int(attempts)
        debug['binned_pairs'] = int(counts.sum())
        debug['nonzero_lags'] = int(np.count_nonzero(counts))
        debug['total_lag_pairs'] = int(counts.sum())
        debug['angle_hist_counts'] = angle_hist.astype(int).tolist()
        if return_debug:
            return np.array([]), np.array([]), np.array([]), debug
        return np.array([]), np.array([]), np.array([])

    gamma_binned = np.full(n_lags, np.nan, dtype=float)
    valid = counts > 0
    gamma_binned[valid] = 0.5 * (sum_sq[valid] / counts[valid])

    debug['attempts'] = int(attempts)
    debug['binned_pairs'] = int(counts.sum())
    debug['nonzero_lags'] = int(np.count_nonzero(counts))
    debug['total_lag_pairs'] = int(counts.sum())
    debug['angle_hist_counts'] = angle_hist.astype(int).tolist()

    if return_debug:
        return bin_centers, gamma_binned, counts, debug
    return bin_centers, gamma_binned, counts


def _deg_to_rad(value):
    return np.radians(value)


def _normalize(vec):
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def az_dip_to_unit_vector(az_deg, dip_deg, dip_positive_down=True):
    """Convert azimuth/dip to a unit vector.

    Args:
        az_deg: azimuth clockwise from North.
        dip_deg: dip from horizontal.
        dip_positive_down: if True, positive dip is down.
    """
    az_rad = _deg_to_rad(az_deg)
    dip_rad = _deg_to_rad(dip_deg)
    sign = 1.0 if dip_positive_down else -1.0
    return _normalize(
        np.array([
            np.sin(az_rad) * np.cos(dip_rad),
            np.cos(az_rad) * np.cos(dip_rad),
            -sign * np.sin(dip_rad),
        ])
    )


def build_orebody_axes(strike_deg, dip_deg, dip_direction_deg=None, dip_positive_down=True):
    """Build orthonormal axes for orebody orientation.

    Args:
        strike_deg: strike azimuth.
        dip_deg: dip angle from horizontal.
        dip_direction_deg: optional dip direction azimuth.
        dip_positive_down: sign convention for dip.

    Returns:
        dict with strike, dip, normal unit vectors (x=east, y=north, z=up)
    """
    strike = az_dip_to_unit_vector(strike_deg, 0.0, dip_positive_down=True)

    if dip_direction_deg is None:
        dip_direction_deg = (strike_deg + 90.0) % 360.0

    dip = az_dip_to_unit_vector(dip_direction_deg, dip_deg, dip_positive_down=dip_positive_down)
    normal = _normalize(np.cross(strike, dip))

    return {'strike': strike, 'dip': dip, 'normal': normal}


def orebody_from_config(config):
    """Return orebody orientation config dict or None."""
    if not config:
        return None
    orebody = config.get('orebody', {})
    if not orebody:
        return None
    return orebody


def apply_variogram_tuning(fitted_model, config):
    tuning = config.get('variogram', {}).get('tuning', {}) if config else {}
    if not tuning.get('enabled'):
        return fitted_model

    range_min = tuning.get('range_min_m')
    range_max = tuning.get('range_max_m')
    target_range = tuning.get('target_range_m')
    nugget_ratio = tuning.get('nugget_ratio')

    if target_range is not None:
        fitted_model.len_scale = float(target_range)
    elif range_min is not None or range_max is not None:
        low = float(range_min) if range_min is not None else fitted_model.len_scale
        high = float(range_max) if range_max is not None else fitted_model.len_scale
        fitted_model.len_scale = float(np.clip(fitted_model.len_scale, low, high))

    if nugget_ratio is not None:
        total_var = fitted_model.nugget + fitted_model.var
        ratio = float(nugget_ratio)
        ratio = np.clip(ratio, 0.0, 0.95)
        fitted_model.nugget = total_var * ratio
        fitted_model.var = total_var * (1.0 - ratio)

    logger.info(
        "Applied variogram tuning: range=%.1f, nugget=%.4f, sill=%.4f",
        fitted_model.len_scale,
        fitted_model.nugget,
        fitted_model.var,
    )
    return fitted_model


def build_covariance_model(model_type='exponential'):
    if model_type == 'exponential':
        return gs.Exponential(dim=3)
    if model_type == 'spherical':
        return gs.Spherical(dim=3)
    if model_type == 'gaussian':
        return gs.Gaussian(dim=3)
    return gs.Exponential(dim=3)


def force_total_sill(model, total_sill: float):
    total_sill = float(total_sill)
    if total_sill <= 0:
        raise ValueError("total sill must be positive")
    total_var = float(model.nugget) + float(model.var)
    if total_var <= 0:
        model.nugget = 0.0
        model.var = total_sill
        return model
    scale = total_sill / total_var
    model.nugget *= scale
    model.var *= scale
    return model


def build_directional_panel_model(base_model, model_type, direction_range):
    panel_model = build_covariance_model(model_type=model_type)
    panel_model.nugget = float(base_model.nugget)
    panel_model.var = float(base_model.var)
    panel_model.len_scale = float(direction_range)
    panel_model.anis = list(getattr(base_model, 'anis', [1.0, 1.0]))
    panel_model.angles = list(getattr(base_model, 'angles', [0.0, 0.0, 0.0]))
    return panel_model




def estimate_variogram(coords, values, n_lags=10, max_dist=200):
    """Estimate experimental variogram."""
    if not GSTOOLS_AVAILABLE:
        raise ImportError("gstools required for variography")

    bin_center, gamma = gs.vario_estimate(
        pos=coords,
        field=values,
        bin_no=n_lags,
        max_dist=max_dist
    )

    return bin_center, gamma


def fit_variogram_model(bins, gamma, model_type='exponential', nugget=True, max_range=2000, total_sill=None):
    """Fit a variogram model to experimental data with bounds."""
    if not GSTOOLS_AVAILABLE:
        raise ImportError("gstools required for variography")

    # Clean data - remove NaN/inf/zero values
    valid_mask = ~np.isnan(gamma) & ~np.isinf(gamma) & (gamma > 0)
    bins_clean = bins[valid_mask]
    gamma_clean = gamma[valid_mask]

    logger.info(f"Valid variogram points: {len(gamma_clean)}/{len(gamma)}")

    if len(gamma_clean) < 3:
        logger.warning("Not enough valid variogram points to fit model")
        model = build_covariance_model(model_type=model_type)
        model.len_scale = 100
        model.nugget = 0.1
        model.var = 0.9
        if total_sill is not None:
            model = force_total_sill(model, total_sill)
        return model

    # Calculate robust sill estimate from experimental variogram
    # Use the asymptote of gamma - usually the last 3 points average
    experimental_sill = np.mean(gamma_clean[-3:]) if len(gamma_clean) >= 3 else np.max(gamma_clean)

    # Select model
    model = build_covariance_model(model_type=model_type)

    max_lag = float(np.nanmax(bins_clean))

    # First attempt: library fit
    try:
        model.fit_variogram(bins_clean, gamma_clean, nugget=nugget)
    except Exception as e:
        logger.warning(f"Variogram fitting failed: {e}; using grid-search fallback")

    def _gamma_curve(h, nug, var, rng):
        h = np.asarray(h, dtype=float)
        rr = max(float(rng), 1e-6)
        if model_type == 'spherical':
            r = h / rr
            core = np.where(
                r < 1.0,
                1.5 * r - 0.5 * (r ** 3),
                1.0,
            )
            return nug + var * core
        if model_type == 'gaussian':
            return nug + var * (1.0 - np.exp(-((h / rr) ** 2)))
        # exponential default
        return nug + var * (1.0 - np.exp(-(h / rr)))

    # Weighted grid-search tightening to improve match in displayed experimental variograms.
    low_lag_boost = 1.0 / np.clip(bins_clean / max(max_lag, 1e-6), 0.05, None)
    weights = low_lag_boost / np.sum(low_lag_boost)

    exp_min = float(np.nanmin(gamma_clean))
    exp_max = float(np.nanmax(gamma_clean))
    sill_ref = max(experimental_sill, exp_max, 1e-3)

    nug_grid = np.linspace(0.0, max(exp_min * 1.5, sill_ref * 0.9), 21) if nugget else np.array([0.0])
    var_grid = np.linspace(max(1e-4, sill_ref * 0.2), max(1e-3, sill_ref * 1.8), 25)
    r_low = max(max_lag / 8.0, 1.0)
    r_high = min(max_range, max_lag * 2.5)
    range_grid = np.linspace(r_low, r_high, 30)

    best = None
    best_err = float("inf")
    for nug_val in nug_grid:
        for var_val in var_grid:
            for range_val in range_grid:
                pred = _gamma_curve(bins_clean, nug_val, var_val, range_val)
                err = float(np.sum(weights * (pred - gamma_clean) ** 2))
                if err < best_err:
                    best_err = err
                    best = (float(nug_val), float(var_val), float(range_val))

    # Prefer tightened solution if it improves the library fit.
    lib_pred = _gamma_curve(
        bins_clean,
        max(float(getattr(model, "nugget", 0.0)), 0.0),
        max(float(getattr(model, "var", 0.0)), 1e-6),
        np.clip(float(getattr(model, "len_scale", max_lag / 2 if max_lag > 0 else 100.0)), 1.0, max_range),
    )
    lib_err = float(np.sum(weights * (lib_pred - gamma_clean) ** 2))

    if best is not None and best_err <= lib_err * 0.995:
        model.nugget, model.var, model.len_scale = best
    else:
        model.nugget = max(float(getattr(model, "nugget", 0.0)), 0.0)
        model.var = max(float(getattr(model, "var", 0.0)), 1e-6)
        model.len_scale = float(np.clip(float(getattr(model, "len_scale", max_lag / 2 if max_lag > 0 else 100.0)), 1.0, max_range))

    # Keep total sill close to experimental asymptote without crushing nugget.
    total_var = model.nugget + model.var
    if total_var > 0:
        scale = sill_ref / total_var
        if 0.7 <= scale <= 1.3:
            model.nugget *= scale
            model.var *= scale
    if total_sill is not None:
        model = force_total_sill(model, total_sill)

    logger.info(f"Fitted {model_type} model: nugget={model.nugget:.4f}, sill={model.var:.4f}, range={model.len_scale:.1f}m")

    return model


def plot_variogram(bins, gamma, model, output_path):
    """Plot variogram."""
    valid = ~np.isnan(gamma) & ~np.isinf(gamma)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(bins[valid], gamma[valid], 'ok', markersize=5, label='Experimental')

    if model is not None:
        x_max = bins[valid].max()
        x_fit = np.linspace(0.1, x_max, 100)  # Start from small non-zero to avoid issues
        # Compute variogram values from the model
        # variogram = nugget + variance * (1 - correlation)
        # Use the correlation function from the model
        y_fit = model.nugget + model.var * (1 - model.cor(x_fit / model.len_scale))
        ax.plot(x_fit, y_fit, 'b-', linewidth=2, label='Fitted Model')

    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Gamma')
    ax.set_title('Variogram')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info(f"Saved variogram plot to {output_path}")


def run(data_path=None, data_dir='data', config=None, output_dir='outputs/figures'):
    """Run variography analysis with directional support."""
    import os

    if not GSTOOLS_AVAILABLE:
        raise RuntimeError("gstools is required for variography")

    # Load data
    if data_path:
        df = pd.read_csv(data_path)
    else:
        from .normal_score import run as run_nst
        _, df = run_nst(data_dir=data_dir)

    # Get coordinates and values
    coords = (df['x'].values, df['y'].values, df['z'].values)
    values = df['tgc_ns'].values

    # Get parameters from config
    n_lags = 10
    max_dist = 200
    model_type = 'exponential'
    directions = []
    normalize_total_sill = False
    target_total_sill = None
    shared_directional_model = False

    max_samples = 2000
    max_pairs = 200000
    if config and 'variogram' in config:
        vario_config = config['variogram']
        n_lags = vario_config.get('n_lags', 10)
        max_dist = vario_config.get('max_distance_m', 200)
        model_type = vario_config.get('model_types', ['exponential'])[0]
        directions = vario_config.get('directions', [])
        max_samples = vario_config.get('max_samples', max_samples)
        max_pairs = vario_config.get('max_pairs', max_pairs)
        normalize_total_sill = bool(vario_config.get('normalize_total_sill', False))
        shared_directional_model = bool(vario_config.get('shared_directional_model', False))
        if normalize_total_sill:
            target_total_sill = float(vario_config.get('total_sill', 1.0))

    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
        coords = (df['x'].values, df['y'].values, df['z'].values)
        values = df['tgc_ns'].values
        logger.info("Subsampled variogram data to %d points", len(df))

    logger.info(f"Computing variogram with {n_lags} lags, max_dist={max_dist}m")

    # Directional variograms from config
    all_bins = {}
    all_gamma = {}
    all_ranges = {}
    all_counts = {}
    all_debug = {}
    all_dir_models = {}

    # Check if directional variograms are configured
    if directions:
        logger.info(f"Computing {len(directions)} directional variograms...")
        for dir_cfg in directions:
            name = dir_cfg.get('name', 'unknown')
            azimuth = dir_cfg.get('azimuth', 0)
            dip = dir_cfg.get('dip', 0)
            tolerance = dir_cfg.get('tolerance', 22.5)

            logger.info(f"  Direction: {name} (az={azimuth}°, dip={dip}°, tol={tolerance}°)")

            orebody = orebody_from_config(config)
            dip_positive_down = True
            if orebody:
                dip_positive_down = orebody.get('dip_positive_down', True)

            bins, gamma, counts, dbg = estimate_directional_variogram(
                coords,
                values,
                azimuth=azimuth,
                dip=dip,
                tolerance=tolerance,
                n_lags=n_lags,
                max_dist=max_dist,
                max_pairs=max_pairs,
                dip_positive_down=dip_positive_down,
                return_debug=True,
            )
            all_debug[name] = {
                'azimuth': float(azimuth),
                'dip': float(dip),
                'tolerance': float(tolerance),
                **dbg,
            }

            if len(bins) > 0:
                all_bins[name] = bins
                all_gamma[name] = gamma
                all_counts[name] = counts
                logger.info(
                    "    Pair diagnostics: binned=%d, nonzero_lags=%d, angle_accept=%.2f%%",
                    int(dbg.get('binned_pairs', 0)),
                    int(dbg.get('nonzero_lags', 0)),
                    100.0 * float(dbg.get('angle_accepted_pairs', 0)) / max(1.0, float(dbg.get('angle_evaluated_pairs', 0))),
                )

                # Fit model for this direction to get range
                dir_model = fit_variogram_model(
                    bins,
                    gamma,
                    model_type=model_type,
                    total_sill=target_total_sill,
                )
                all_ranges[name] = dir_model.len_scale
                all_dir_models[name] = dir_model
                logger.info(f"    Range: {dir_model.len_scale:.1f}m, nugget={dir_model.nugget:.4f}, sill={dir_model.var:.4f}")

    # Always compute omnidirectional as fallback/comparison
    logger.info("Computing omnidirectional variogram...")
    omni_bins, omni_gamma = estimate_variogram(coords, values, n_lags=n_lags, max_dist=max_dist)
    all_bins['omnidirectional'] = omni_bins
    all_gamma['omnidirectional'] = omni_gamma

    # Fit final model - use omnidirectional or create anisotropic
    # Filter out unreasonable ranges (>2000m) before computing anisotropy
    max_reasonable_range = 2000  # 2km max for deposit scale

    valid_ranges = {k: v for k, v in all_ranges.items() if v <= max_reasonable_range}
    logger.info(f"Valid ranges (≤{max_reasonable_range}m): {valid_ranges}")

    if len(valid_ranges) >= 2:
        logger.info("Creating anisotropic variogram model...")

        # Prefer config-provided ranges for anisotropy if available
        anis_cfg = vario_config.get('anisotropy', {}) if config and 'variogram' in config else {}
        ranges_cfg = anis_cfg.get('ranges_m', {})
        if ranges_cfg:
            strike_range = ranges_cfg.get('strike')
            dip_range = ranges_cfg.get('down_dip')
            normal_range = ranges_cfg.get('normal')
        else:
            strike_range = valid_ranges.get('along_strike') or valid_ranges.get('strike')
            dip_range = valid_ranges.get('down_dip')
            normal_range = valid_ranges.get('normal_to_plane') or valid_ranges.get('across_strike')

        ranges_list = [r for r in [strike_range, dip_range, normal_range] if r]
        if len(ranges_list) < 2:
            fitted_model = fit_variogram_model(
                omni_bins,
                omni_gamma,
                model_type=model_type,
                total_sill=target_total_sill,
            )
            logger.info("Insufficient directional ranges; using omnidirectional variogram model")
        else:
            major_range = max(ranges_list)
            anis_y = strike_range / dip_range if strike_range and dip_range else 1.0
            anis_z = strike_range / normal_range if strike_range and normal_range else 1.0

            fitted_model = fit_variogram_model(
                omni_bins,
                omni_gamma,
                model_type=model_type,
                total_sill=target_total_sill,
            )
            fitted_model.len_scale = major_range
            fitted_model.anis = [anis_y, anis_z]

            # Apply geological orientation if provided
            orebody = orebody_from_config(config)
            if orebody:
                strike_deg = orebody.get('strike_deg')
                dip_deg = orebody.get('dip_deg')
                dip_direction_deg = orebody.get('dip_direction_deg')
                dip_positive_down = orebody.get('dip_positive_down', True)
                if strike_deg is not None and dip_deg is not None:
                    axes = build_orebody_axes(
                        float(strike_deg),
                        float(dip_deg),
                        float(dip_direction_deg) if dip_direction_deg is not None else None,
                        dip_positive_down=bool(dip_positive_down),
                    )
                    fitted_model.angles = [
                        np.radians(float(strike_deg)),
                        np.radians(float(dip_deg)),
                        0.0,
                    ]
                    logger.info(
                        "Applied model rotation: strike=%.1f°, dip=%.1f°",
                        float(strike_deg), float(dip_deg)
                    )

            logger.info(
                "Anisotropic model: range=%s m, anis=[%.2f, %.2f]",
                f"{major_range:.1f}", anis_y, anis_z
            )
    else:
        fitted_model = fit_variogram_model(
            omni_bins,
            omni_gamma,
            model_type=model_type,
            total_sill=target_total_sill,
        )
        logger.info("Using omnidirectional variogram model")

    if target_total_sill is not None:
        fitted_model = force_total_sill(fitted_model, target_total_sill)

    panel_models = {}
    if shared_directional_model:
        directional_ranges = {}
        if config and 'variogram' in config:
            ranges_cfg = config['variogram'].get('anisotropy', {}).get('ranges_m', {})
            directional_ranges = {
                'along_strike': ranges_cfg.get('strike'),
                'strike': ranges_cfg.get('strike'),
                'down_dip': ranges_cfg.get('down_dip'),
                'normal_to_plane': ranges_cfg.get('normal'),
                'across_strike': ranges_cfg.get('normal'),
            }
        for name, direction_range in all_ranges.items():
            use_range = directional_ranges.get(name) or direction_range
            if use_range:
                panel_models[name] = build_directional_panel_model(fitted_model, model_type, use_range)

    # Plot all directions
    os.makedirs(output_dir, exist_ok=True)

    if len(all_bins) > 1:
        # Multi-panel plot
        n_dirs = len(all_bins)
        n_cols = min(3, n_dirs)
        n_rows = (n_dirs + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        if n_dirs == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for idx, (name, bins) in enumerate(all_bins.items()):
            ax = axes[idx]
            gamma = all_gamma[name]

            valid = ~np.isnan(gamma) & ~np.isinf(gamma) & (gamma > 0)
            ax.plot(bins[valid], gamma[valid], 'ok', markersize=5, label='Experimental')

            # Use a shared final nugget/sill treatment when requested so the
            # directional panels only differ by geometry and range.
            panel_model = panel_models.get(name) or all_dir_models.get(name, fitted_model)
            x_max = bins[valid].max() if valid.any() else max_dist
            x_fit = np.linspace(0.1, x_max, 100)
            y_fit = panel_model.nugget + panel_model.var * (1 - panel_model.cor(x_fit / panel_model.len_scale))
            ax.plot(x_fit, y_fit, 'b-', linewidth=2, label='Model')

            ax.set_xlabel('Distance (m)')
            ax.set_ylabel('Gamma')
            ax.set_title(f'Variogram: {name}')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Hide empty subplots
        for idx in range(n_dirs, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'variogram.png')
    else:
        # Single panel
        fig, ax = plt.subplots(figsize=(8, 5))
        valid = ~np.isnan(omni_gamma) & ~np.isinf(omni_gamma)
        ax.plot(omni_bins[valid], omni_gamma[valid], 'ok', markersize=5, label='Experimental')

        if fitted_model is not None:
            x_max = omni_bins[valid].max()
            x_fit = np.linspace(0.1, x_max, 100)
            y_fit = fitted_model.nugget + fitted_model.var * (1 - fitted_model.cor(x_fit / fitted_model.len_scale))
            ax.plot(x_fit, y_fit, 'b-', linewidth=2, label='Fitted Model')

        ax.set_xlabel('Distance (m)')
        ax.set_ylabel('Gamma')
        ax.set_title('Variogram')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = os.path.join(output_dir, 'variogram.png')

    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.info(f"Saved variogram plot to {plot_path}")

    ranges = {'omnidirectional': fitted_model.len_scale}
    ranges.update(all_ranges)
    logger.info(f"Variogram ranges: {ranges}")

    # Final validation - ensure sill is reasonable for SGS
    if fitted_model.var < 0.1:
        logger.warning(f"Variogram sill ({fitted_model.var:.4f}) too low - setting to 1.0")
        fitted_model.var = 1.0

    if fitted_model.nugget + fitted_model.var < 0.5:
        total_var = fitted_model.nugget + fitted_model.var
        logger.warning(f"Total variance ({total_var:.4f}) too low - normalizing")
        scale = 1.0 / total_var
        fitted_model.var *= scale
        fitted_model.nugget *= scale

    fitted_model = apply_variogram_tuning(fitted_model, config)
    if target_total_sill is not None:
        fitted_model = force_total_sill(fitted_model, target_total_sill)

    logger.info(
        "Final variogram: nugget=%.4f, sill=%.4f, total_sill=%.4f, range=%.1fm",
        fitted_model.nugget,
        fitted_model.var,
        fitted_model.nugget + fitted_model.var,
        fitted_model.len_scale,
    )

    # Persist model parameters for paper tables
    model_meta = {
        'model_type': model_type,
        'nugget': float(fitted_model.nugget),
        'sill': float(fitted_model.var),
        'total_sill': float(fitted_model.nugget + fitted_model.var),
        'len_scale': float(fitted_model.len_scale),
        'anis': [float(v) for v in getattr(fitted_model, 'anis', [1.0, 1.0])],
        'angles': [float(v) for v in getattr(fitted_model, 'angles', [0.0, 0.0, 0.0])],
        'direction_ranges': {k: float(v) for k, v in all_ranges.items()},
        'shared_directional_model': bool(shared_directional_model),
        'nested_structures': ((config or {}).get('variogram', {}) or {}).get('nested_structures', {}),
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'variogram_model.json'), 'w') as f:
        import json
        json.dump(model_meta, f, indent=2)

    # Save pair counts per direction
    if all_counts:
        with open(os.path.join(output_dir, 'variogram_pair_counts.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['direction', 'lag', 'count'])
            for name, counts in all_counts.items():
                for i, count in enumerate(counts):
                    writer.writerow([name, i + 1, int(count)])

    if all_debug:
        debug_path = os.path.join(output_dir, 'variogram_direction_debug.csv')
        with open(debug_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'direction', 'azimuth_deg', 'dip_deg', 'tolerance_deg',
                'n_points', 'possible_pairs', 'target_pairs',
                'attempts', 'non_self_pairs', 'distance_filtered_pairs',
                'angle_evaluated_pairs', 'angle_accepted_pairs', 'binned_pairs',
                'nonzero_lags', 'total_lag_pairs',
                'angle_acceptance_pct', 'binned_from_angle_pct',
            ])
            for name, dbg in all_debug.items():
                angle_eval = float(dbg.get('angle_evaluated_pairs', 0))
                angle_ok = float(dbg.get('angle_accepted_pairs', 0))
                binned = float(dbg.get('binned_pairs', 0))
                writer.writerow([
                    name, dbg.get('azimuth'), dbg.get('dip'), dbg.get('tolerance'),
                    dbg.get('n_points'), dbg.get('possible_pairs'), dbg.get('target_pairs'),
                    dbg.get('attempts'), dbg.get('non_self_pairs'), dbg.get('distance_filtered_pairs'),
                    dbg.get('angle_evaluated_pairs'), dbg.get('angle_accepted_pairs'), dbg.get('binned_pairs'),
                    dbg.get('nonzero_lags'), dbg.get('total_lag_pairs'),
                    100.0 * angle_ok / max(1.0, angle_eval),
                    100.0 * binned / max(1.0, angle_ok),
                ])
        logger.info("Saved directional debug summary: %s", debug_path)

        lag_occ_path = os.path.join(output_dir, 'variogram_lag_occupancy.csv')
        with open(lag_occ_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['direction', 'lag', 'count', 'pct_of_direction_pairs'])
            for name, counts in all_counts.items():
                total = float(np.sum(counts))
                for i, count in enumerate(counts):
                    writer.writerow([name, i + 1, int(count), 100.0 * float(count) / max(1.0, total)])
        logger.info("Saved lag occupancy diagnostics: %s", lag_occ_path)

        angle_hist_path = os.path.join(output_dir, 'variogram_angle_histogram.csv')
        with open(angle_hist_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['direction', 'angle_bin_start_deg', 'angle_bin_end_deg', 'count', 'pct_of_angle_evaluated'])
            for name, dbg in all_debug.items():
                bins = dbg.get('angle_hist_bins_deg', [])
                counts = dbg.get('angle_hist_counts', [])
                total = float(sum(counts))
                if len(bins) < 2:
                    continue
                for i, count in enumerate(counts):
                    writer.writerow([name, bins[i], bins[i + 1], int(count), 100.0 * float(count) / max(1.0, total)])
        logger.info("Saved angle histogram diagnostics: %s", angle_hist_path)

    # Return model in a format compatible with SGS
    return fitted_model, {'bins': all_bins, 'gamma': all_gamma}, ranges
