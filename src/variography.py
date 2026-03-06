"""
variography.py - Variography

Computes directional variograms and fits anisotropic models.
Supports omnidirectional and directional variograms with anisotropy.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

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
        bin_centers, gamma values
    """
    x, y, z = coords

    # Convert azimuth/dip to radians
    az_rad = np.radians(azimuth)
    dip_rad = np.radians(dip)

    # Compute direction vector (dip-positive-down)
    dir_vec = az_dip_to_unit_vector(azimuth, dip, dip_positive_down=dip_positive_down)
    dir_x, dir_y, dir_z = dir_vec

    # Compute distances and projections along direction
    n = len(x)
    distances = []
    gamma_vals = []

    pair_count = 0
    order = np.random.permutation(n)
    for i in order:
        for j in order:
            # Distance between points
            dx = x[j] - x[i]
            dy = y[j] - y[i]
            dz_val = z[j] - z[i]
            h = np.sqrt(dx**2 + dy**2 + dz_val**2)

            if h < 1 or h > max_dist:
                continue

            # Project onto direction vector
            proj = dx * dir_x + dy * dir_y + dz_val * dir_z

            # Check if within angular tolerance
            if h > 0:
                cos_angle = abs(proj) / h
                angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

                if angle <= tolerance:
                    distances.append(abs(proj))
                    gamma_vals.append((values[i] - values[j])**2)
                    pair_count += 1
                    if pair_count >= max_pairs:
                        break
        if pair_count >= max_pairs:
            break

    if len(gamma_vals) < 10:
        logger.warning(f"Too few pairs for direction {azimuth}/{dip}: {len(gamma_vals)}")
        return np.array([]), np.array([])

    # Bin the distances
    bins = np.linspace(0, max_dist, n_lags + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    gamma_binned = np.zeros(n_lags)

    for d, g in zip(distances, gamma_vals):
        bin_idx = np.searchsorted(bins[1:], d)
        if bin_idx < n_lags:
            gamma_binned[bin_idx] += g

    # Count pairs per bin
    counts = np.zeros(n_lags)
    for d in distances:
        bin_idx = np.searchsorted(bins[1:], d)
        if bin_idx < n_lags:
            counts[bin_idx] += 1

    # Normalize
    valid = counts > 0
    gamma_binned[valid] /= (2 * counts[valid])

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


def fit_variogram_model(bins, gamma, model_type='exponential', nugget=True, max_range=2000):
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
        model = gs.Exponential(dim=3)
        model.len_scale = 100
        model.nugget = 0.1
        model.var = 0.9
        return model

    # Calculate robust sill estimate from experimental variogram
    # Use the asymptote of gamma - usually the last 3 points average
    experimental_sill = np.mean(gamma_clean[-3:]) if len(gamma_clean) >= 3 else np.max(gamma_clean)

    # Select model
    if model_type == 'exponential':
        model = gs.Exponential(dim=3)
    elif model_type == 'spherical':
        model = gs.Spherical(dim=3)
    elif model_type == 'gaussian':
        model = gs.Gaussian(dim=3)
    else:
        model = gs.Exponential(dim=3)

    # Fit with bounds to prevent unrealistic ranges
    try:
        model.fit_variogram(bins_clean, gamma_clean, nugget=nugget)

        # Validate and fix parameters
        # Range check
        max_lag = np.nanmax(bins_clean)
        if model.len_scale <= 0 or model.len_scale > max_range or model.len_scale > max_lag * 3:
            logger.warning(f"Range {model.len_scale:.1f}m invalid, setting to {max_lag/2:.1f}m")
            model.len_scale = max_lag / 2

        # Nugget check - should be reasonable (0 to ~50% of sill)
        if model.nugget < 0:
            model.nugget = 0
        if model.nugget > experimental_sill * 0.8:
            logger.warning(f"Nugget {model.nugget:.4f} too high, reducing to 30% of sill")
            model.nugget = experimental_sill * 0.3

        # Sill check - must be positive and = total variance - nugget
        if model.var <= 0:
            model.var = experimental_sill - model.nugget
        if model.var <= 0:
            model.var = 0.5  # Default if sill is still invalid

        # Total variance check
        total_var = model.nugget + model.var
        if abs(total_var - experimental_sill) > experimental_sill * 0.5:
            # Recalibrate to match experimental sill
            scale_factor = experimental_sill / total_var if total_var > 0 else 1.0
            model.var = model.var * scale_factor
            model.nugget = model.nugget * scale_factor

        logger.info(f"Fitted {model_type} model: nugget={model.nugget:.4f}, sill={model.var:.4f}, range={model.len_scale:.1f}m")
    except Exception as e:
        logger.warning(f"Variogram fitting failed: {e}, using robust defaults")
        model.len_scale = max_lag / 2 if max_lag > 0 else 100
        model.nugget = experimental_sill * 0.3
        model.var = experimental_sill * 0.7

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

            bins, gamma, counts = estimate_directional_variogram(
                coords,
                values,
                azimuth=azimuth,
                dip=dip,
                tolerance=tolerance,
                n_lags=n_lags,
                max_dist=max_dist,
                max_pairs=max_pairs,
                dip_positive_down=dip_positive_down,
            )

            if len(bins) > 0:
                all_bins[name] = bins
                all_gamma[name] = gamma
                all_counts[name] = counts

                # Fit model for this direction to get range
                dir_model = fit_variogram_model(bins, gamma, model_type=model_type)
                all_ranges[name] = dir_model.len_scale
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
            fitted_model = fit_variogram_model(omni_bins, omni_gamma, model_type=model_type)
            logger.info("Insufficient directional ranges; using omnidirectional variogram model")
        else:
            major_range = max(ranges_list)
            anis_y = strike_range / dip_range if strike_range and dip_range else 1.0
            anis_z = strike_range / normal_range if strike_range and normal_range else 1.0

            fitted_model = fit_variogram_model(omni_bins, omni_gamma, model_type=model_type)
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
        fitted_model = fit_variogram_model(omni_bins, omni_gamma, model_type=model_type)
        logger.info("Using omnidirectional variogram model")

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

            # Plot model
            x_max = bins[valid].max() if valid.any() else max_dist
            x_fit = np.linspace(0.1, x_max, 100)
            y_fit = fitted_model.nugget + fitted_model.var * (1 - fitted_model.cor(x_fit / fitted_model.len_scale))
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

    logger.info(f"Final variogram: nugget={fitted_model.nugget:.4f}, sill={fitted_model.var:.4f}, range={fitted_model.len_scale:.1f}m")

    # Persist model parameters for paper tables
    model_meta = {
        'model_type': model_type,
        'nugget': float(fitted_model.nugget),
        'sill': float(fitted_model.var),
        'len_scale': float(fitted_model.len_scale),
        'anis': [float(v) for v in getattr(fitted_model, 'anis', [1.0, 1.0])],
        'angles': [float(v) for v in getattr(fitted_model, 'angles', [0.0, 0.0, 0.0])],
        'direction_ranges': {k: float(v) for k, v in all_ranges.items()}
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'variogram_model.json'), 'w') as f:
        import json
        json.dump(model_meta, f, indent=2)

    # Save pair counts per direction
    if all_counts:
        import csv
        with open(os.path.join(output_dir, 'variogram_pair_counts.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['direction', 'lag', 'count'])
            for name, counts in all_counts.items():
                for i, count in enumerate(counts):
                    writer.writerow([name, i + 1, int(count)])

    # Return model in a format compatible with SGS
    return fitted_model, {'bins': all_bins, 'gamma': all_gamma}, ranges
