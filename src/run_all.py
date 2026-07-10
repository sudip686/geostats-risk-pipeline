"""
run_all.py - Main CLI Entrypoint

Runs the complete SGS workflow from configuration.
"""

import argparse
import copy
import logging
import os
import subprocess
import sys
import json
import time
import shutil
from contextlib import contextmanager

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def _pid_exists(pid):
    if pid <= 0:
        return False
    if sys.platform.startswith("win"):
        proc = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        line = (proc.stdout or "").strip()
        return bool(line) and not line.startswith("INFO:")
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


@contextmanager
def _output_run_lock(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    lock_path = os.path.join(output_dir, ".run_all.lock.json")
    payload = {
        "pid": os.getpid(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "output_dir": os.path.abspath(output_dir),
    }
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            break
        except FileExistsError:
            try:
                with open(lock_path, "r", encoding="utf-8-sig") as f:
                    existing = json.load(f)
                existing_pid = int(existing.get("pid", 0))
            except Exception:
                existing_pid = 0
            if _pid_exists(existing_pid):
                raise RuntimeError(
                    f"Output directory is already locked by live PID {existing_pid}: {lock_path}"
                )
            logger.warning("Removing stale run lock: %s", lock_path)
            try:
                os.remove(lock_path)
            except FileNotFoundError:
                pass

    try:
        yield
    finally:
        try:
            with open(lock_path, "r", encoding="utf-8-sig") as f:
                current = json.load(f)
            if int(current.get("pid", 0)) == os.getpid():
                os.remove(lock_path)
        except FileNotFoundError:
            pass
        except Exception as exc:
            logger.warning("Could not remove run lock %s: %s", lock_path, exc)


def _abs_repo_path(rel_path):
    return os.path.normcase(os.path.abspath(os.path.join(REPO_ROOT, rel_path)))


def _azimuth_diff_deg(a, b):
    return abs(((float(a) - float(b) + 90.0) % 180.0) - 90.0)


def _validate_canonical_workflow_contract(config_path, output_dir, config, profile_name=None):
    contract = (config or {}).get('workflow_contract', {}) or {}
    if not contract:
        return

    canonical_cfg = str(contract.get('canonical_config_path', 'config/main_config.yaml'))
    canonical_out = str(contract.get('canonical_output_dir', 'output/a3_categorical_25_50_nr100'))
    require_paths = bool(contract.get('require_canonical_paths', True))

    resolved_cfg = os.path.normcase(os.path.abspath(config_path))
    resolved_out = os.path.normcase(os.path.abspath(output_dir))
    expected_cfg = _abs_repo_path(canonical_cfg)
    expected_out = _abs_repo_path(canonical_out)

    if profile_name:
        if require_paths and resolved_cfg != expected_cfg:
            raise ValueError(
                f"Runtime profile '{profile_name}' must start from the canonical config path {canonical_cfg}."
            )
        if resolved_out == expected_out:
            raise ValueError(
                f"Runtime profile '{profile_name}' cannot write to the canonical output directory {canonical_out}. "
                "Choose a separate exploratory output path."
            )
        logger.warning(
            "Runtime profile '%s' active; allowing non-canonical output directory %s while preserving the canonical config path.",
            profile_name,
            output_dir,
        )
        return

    allow_env = str(contract.get('allow_noncanonical_with_env', '') or '').strip()
    if allow_env and os.environ.get(allow_env, '').lower() in {'1', 'true', 'yes'}:
        logger.warning("Non-canonical workflow override enabled via %s", allow_env)
        return

    if require_paths and resolved_cfg != expected_cfg:
        raise ValueError(
            f"Non-canonical config path rejected: {config_path}. Use {canonical_cfg} for production runs."
        )
    if require_paths and resolved_out != expected_out:
        raise ValueError(
            f"Non-canonical output path rejected: {output_dir}. Use {canonical_out} for production runs."
        )

    domains_cfg = (config.get('domains', {}) or {})
    sim_cfg = (config.get('simulation', {}) or {})
    vario_cfg = (config.get('variogram', {}) or {})
    report_cfg = (config.get('reporting_grid', {}) or {})
    grid_cfg = (config.get('grid', {}) or {})
    calib_cfg = (config.get('calibration', {}) or {})
    internal_cfg = (config.get('internal_validation', {}) or {})
    orebody_cfg = (config.get('orebody', {}) or {})
    directions = {
        str(item.get('name')): item
        for item in (vario_cfg.get('directions') or [])
        if isinstance(item, dict) and item.get('name')
    }

    def require(condition, message):
        if not condition:
            raise ValueError(message)

    require(bool(domains_cfg.get('hard_boundaries')), "Canonical workflow requires geology-led hard boundaries.")
    require(bool(domains_cfg.get('categorical_simulation')), "Canonical workflow requires stochastic categorical domaining.")

    groups = domains_cfg.get('canonical_groups', {}) or {}
    require(
        {'fresh_graphitic', 'weathered_graphitic', 'host_waste'}.issubset(set(groups.keys())),
        "Canonical workflow requires fresh_graphitic, weathered_graphitic, and host_waste categories.",
    )

    require(float(grid_cfg.get('dx', 0)) == 25.0 and float(grid_cfg.get('dy', 0)) == 25.0 and float(grid_cfg.get('dz', 0)) == 2.0,
            "Canonical simulation support must be 25 x 25 x 2 m.")
    require(float(report_cfg.get('dx', 0)) == 50.0 and float(report_cfg.get('dy', 0)) == 50.0 and float(report_cfg.get('dz', 0)) == 2.0,
            "Canonical reporting support must be 50 x 50 x 2 m.")
    require(list(grid_cfg.get('origin_xyz', [])) == list(report_cfg.get('origin_xyz', [])),
            "Simulation and reporting grids must share the same origin.")

    require(bool(vario_cfg.get('normalize_total_sill')), "Canonical SGS requires variogram.normalize_total_sill = true.")
    require(float(vario_cfg.get('total_sill', 0.0)) == 1.0, "Canonical SGS requires variogram.total_sill = 1.0.")
    require(bool(vario_cfg.get('shared_directional_model')), "Canonical variogram reporting requires one nugget interpretation across directional panels.")
    nested_cfg = vario_cfg.get('nested_structures', {}) or {}
    require('enabled' in nested_cfg, "Canonical workflow must declare whether nested variogram structures are enabled.")
    require(not bool(nested_cfg.get('enabled')), "Canonical production workflow currently permits only the explicit single-structure model.")
    require({'along_strike', 'down_dip', 'normal_to_plane'}.issubset(set(directions.keys())),
            "Canonical workflow requires along_strike, down_dip, and normal_to_plane variogram directions.")
    require(_azimuth_diff_deg(directions['along_strike'].get('azimuth', 0.0), orebody_cfg.get('strike_deg', 0.0)) <= 1.0,
            "along_strike variogram azimuth must align with orebody strike.")
    require(abs(float(directions['along_strike'].get('dip', 999.0))) <= 1.0,
            "along_strike variogram dip must be subhorizontal.")
    require(_azimuth_diff_deg(directions['down_dip'].get('azimuth', 0.0), orebody_cfg.get('dip_direction_deg', 0.0)) <= 1.0,
            "down_dip variogram azimuth must align with orebody dip direction.")
    require(abs(float(directions['down_dip'].get('dip', 999.0)) - float(orebody_cfg.get('dip_deg', 0.0))) <= 1.0,
            "down_dip variogram dip must align with orebody dip.")

    require(list(sim_cfg.get('search_radius_m', [])) == list(domains_cfg.get('search_radius_m', [])),
            "Categorical and grade SGS search radii must match in the canonical workflow.")
    require(bool(sim_cfg.get('require_full_neighborhood')), "Canonical SGS requires a fixed neighborhood policy.")
    require(int(sim_cfg.get('n_real', 0)) == 100, "Canonical workflow requires 100 realizations.")

    require(not bool(calib_cfg.get('enabled')), "Calibration must remain disabled in the canonical workflow.")
    require(str(calib_cfg.get('reference_data', '') or '').strip() == '', "Calibration reference_data must be blank in the canonical workflow.")
    require(not bool(internal_cfg.get('enabled')), "Internal validation against external reference models must remain disabled.")
    require(str(internal_cfg.get('model_csv', '') or '').strip() == '', "internal_validation.model_csv must be blank in the canonical workflow.")

    benchmark_cfg = (config.get('classification_benchmark', {}) or {})
    require(bool(benchmark_cfg.get('enabled')), "Canonical workflow requires the 15%-at-90%-confidence benchmark diagnostic.")
    require(abs(float(benchmark_cfg.get('relative_error_limit', 0.0)) - 0.15) <= 1e-9,
            "Canonical workflow requires a 15% relative error benchmark.")
    require(abs(float(benchmark_cfg.get('confidence_interval', 0.0)) - 0.90) <= 1e-9,
            "Canonical workflow requires a 90% confidence benchmark.")


def _clone_config(config):
    return copy.deepcopy(config)


def _save_sgs_outputs(result, config, output_dir, n_real, seed):
    from src.utils.io import save_grid

    os.makedirs(output_dir, exist_ok=True)
    save_grid(result, output_dir, prefix='sgs')
    if 'realizations_ns' in result:
        np.save(os.path.join(output_dir, 'sgs_reals_ns.npy'), result['realizations_ns'])

    meta = {
        'dx': result['grid_def']['dx'],
        'dy': result['grid_def']['dy'],
        'dz': result['grid_def']['dz'],
        'nx': result['grid_def']['nx'],
        'ny': result['grid_def']['ny'],
        'nz': result['grid_def']['nz'],
        'x_min': float(result['x'][0]),
        'x_max': float(result['x'][-1]),
        'y_min': float(result['y'][0]),
        'y_max': float(result['y'][-1]),
        'z_min': float(result['z'][0]),
        'z_max': float(result['z'][-1]),
        'n_realizations': int(n_real),
        'seed': int(seed),
        'orebody': config.get('orebody', {}) if config else {},
    }
    with open(os.path.join(output_dir, 'sgs_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)


def _run_hard_domain_branch(name, payload, config, grid_mask, output_root, grade_field):
    from src import declustering
    from src import normal_score
    from src import sgs
    from src import variography

    branch_cfg = _clone_config(config)
    branch_cfg['target_lith_codes'] = list(payload['lith_codes'])
    if payload.get('search_radius_m'):
        branch_cfg.setdefault('simulation', {})
        branch_cfg['simulation']['search_radius_m'] = list(payload['search_radius_m'])

    branch_dir = os.path.join(output_root, 'domains', name)
    grids_dir = os.path.join(branch_dir, 'grids')
    figures_dir = os.path.join(branch_dir, 'figures')
    tables_dir = os.path.join(branch_dir, 'tables')
    os.makedirs(grids_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    domain_df = payload['data'].copy()
    if domain_df.empty:
        raise RuntimeError(f"Domain '{name}' has no conditioning samples")
    domain_path = os.path.join(branch_dir, 'domain_data.csv')
    domain_df.to_csv(domain_path, index=False)

    declustered, dc_stats = declustering.decluster_data(
        domain_df,
        cell_size_xy=config.get('declustering', {}).get('cell_size_xy_m', 200),
        cell_size_z=config.get('declustering', {}).get('cell_size_z_m', 5),
        grade_field=grade_field,
    )
    declustered_path = os.path.join(branch_dir, 'declustered.csv')
    declustered.to_csv(declustered_path, index=False)

    nst, nst_data = normal_score.run(
        data_path=declustered_path,
        grade_field=grade_field,
        output_path=os.path.join(branch_dir, 'nst_data.csv'),
        config=branch_cfg,
    )
    nst.save(os.path.join(branch_dir, 'nst_params.json'))

    vario_model, _exp_variograms, ranges = variography.run(
        data_path=os.path.join(branch_dir, 'nst_data.csv'),
        config=branch_cfg,
        output_dir=figures_dir,
    )

    sim_cfg = branch_cfg.get('simulation', {}) or {}
    n_real = int(sim_cfg.get('n_real', 100))
    seed = int(sim_cfg.get('seed', 1337))
    grid_def = sgs.define_grid(branch_cfg, nst_data)
    validation_warnings = sgs.validate_variogram_for_sgs(vario_model, grid_def, branch_cfg)
    for warning in validation_warnings:
        if 'ERROR' in warning:
            logger.error("[%s] %s", name, warning)
        else:
            logger.warning("[%s] %s", name, warning)
    if any('ERROR' in warning for warning in validation_warnings):
        raise RuntimeError(f"Invalid variogram parameters for domain '{name}'")

    requested_n_jobs = int(sim_cfg.get('n_jobs', -1))
    n_jobs = requested_n_jobs
    if n_jobs == -1 and sgs.JOBLIB_AVAILABLE:
        import multiprocessing
        n_jobs = min(multiprocessing.cpu_count(), n_real)
    n_cells = grid_def['nx'] * grid_def['ny'] * grid_def['nz']
    if n_cells >= 200_000 and requested_n_jobs == -1:
        n_jobs = min(n_jobs, 4)
    chunk_size = sim_cfg.get('krige_chunk_size', 5000)
    if chunk_size is not None and chunk_size <= 0:
        chunk_size = None

    result = sgs.run_sgs(
        nst_data,
        grid_def,
        vario_model,
        nst,
        n_realizations=n_real,
        seed=seed,
        n_jobs=n_jobs,
        chunk_size=chunk_size,
        search_radius=sim_cfg.get('search_radius_m'),
        max_neighbors=sim_cfg.get('max_neighbors'),
        min_neighbors=sim_cfg.get('min_neighbors'),
        update_every=int(sim_cfg.get('local_update_every', 1)),
        config=branch_cfg,
        output_dir=grids_dir,
        grid_mask=grid_mask,
        require_full_neighborhood=bool(sim_cfg.get('require_full_neighborhood', False)),
        checkpoint_every=int(sim_cfg.get('checkpoint_every', 5000)),
    )
    _save_sgs_outputs(result, branch_cfg, grids_dir, n_real=n_real, seed=seed)

    return {
        'name': name,
        'config': branch_cfg,
        'domain_path': domain_path,
        'decluster_stats': dc_stats,
        'ranges': ranges,
        'result': result,
        'lith_codes': list(payload['lith_codes']),
        'n_samples': int(len(domain_df)),
        'figures_dir': figures_dir,
    }


def _mosaic_domain_results(domain_runs, domain_masks):
    ordered = list(domain_runs.keys())
    base = domain_runs[ordered[0]]['result']
    reals = np.full_like(base['realizations'], np.nan)
    reals_ns = np.full_like(base['realizations_ns'], np.nan)

    for name in ordered:
        mask = np.asarray(domain_masks[name], dtype=bool)
        reals[:, mask] = domain_runs[name]['result']['realizations'][:, mask]
        reals_ns[:, mask] = domain_runs[name]['result']['realizations_ns'][:, mask]

    if np.isnan(reals).any() or np.isnan(reals_ns).any():
        raise RuntimeError("Domain mosaicking left unassigned cells in the canonical realization stack")

    return {
        'realizations': reals.astype(np.float32),
        'realizations_ns': reals_ns.astype(np.float32),
        'x': base['x'],
        'y': base['y'],
        'z': base['z'],
        'grid_def': base['grid_def'],
        'timing': {
            'total_seconds': float(sum(run['result']['timing']['total_seconds'] for run in domain_runs.values())),
            'avg_per_real': float(np.mean([run['result']['timing']['avg_per_real'] for run in domain_runs.values()])),
        },
    }


def _write_top_level_domain_variogram(domain_runs, output_dir):
    primary_name = 'fresh_graphitic' if 'fresh_graphitic' in domain_runs else next(iter(domain_runs))
    primary_fig = os.path.join(domain_runs[primary_name]['figures_dir'], 'variogram.png')
    primary_json = os.path.join(domain_runs[primary_name]['figures_dir'], 'variogram_model.json')
    if os.path.exists(primary_fig):
        shutil.copy2(primary_fig, os.path.join(output_dir, 'figures', 'variogram.png'))
    if os.path.exists(primary_json):
        payload = json.loads(open(primary_json, 'r', encoding='utf-8').read())
    else:
        payload = {}
    payload['primary_domain'] = primary_name
    payload['per_domain'] = {
        name: {
            'lith_codes': run['lith_codes'],
            'n_samples': run['n_samples'],
            'ranges_m': {k: float(v) for k, v in run['ranges'].items()},
        }
        for name, run in domain_runs.items()
    }
    with open(os.path.join(output_dir, 'figures', 'variogram_model.json'), 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)

    pair_counts = os.path.join(domain_runs[primary_name]['figures_dir'], 'variogram_pair_counts.csv')
    if os.path.exists(pair_counts):
        shutil.copy2(pair_counts, os.path.join(output_dir, 'figures', 'variogram_pair_counts.csv'))


def run_full_workflow(config_path='config/main_config.yaml', output_dir='outputs', profile_name=None):
    """
    Run the complete SGS workflow.

    Args:
        config_path: Path to YAML configuration
        output_dir: Output directory
    """
    from src.utils.io import PROFILE_ENV_VAR, load_config

    selected_profile = str(profile_name or os.environ.get(PROFILE_ENV_VAR, '') or '').strip() or None
    if selected_profile:
        os.environ[PROFILE_ENV_VAR] = selected_profile

    # Load config
    logger.info(f"Loading configuration from {config_path}")
    config = load_config(config_path, profile_name=selected_profile)
    _validate_canonical_workflow_contract(config_path, output_dir, config, profile_name=selected_profile)
    if selected_profile:
        logger.info("Runtime profile '%s' is active for this workflow run", selected_profile)

    if os.environ.get('CI', '').lower() in {'1', 'true', 'yes'}:
        config = _clone_config(config)
        sim = dict(config.get('simulation', {}))
        sim['n_real'] = config.get('ci', {}).get('n_real', sim.get('n_real', 100))
        config['simulation'] = sim

    # Get grade field from config
    grade_field = config.get('grade_field', 'tgc_pct')

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'grids'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'tables'), exist_ok=True)

    timings = {}

    # ============ Step 0: Validation ============
    logger.info("=" * 50)
    logger.info("STEP 0: Input Validation")
    logger.info("=" * 50)
    from src import validate_inputs
    start = time.time()
    validation = validate_inputs.run_validation(
        data_dir=config.get('data_dir', 'data'),
        config_path=config_path,
        output_dir=output_dir,
    )
    timings['validation_seconds'] = time.time() - start
    if not validation['passed']:
        logger.error("Validation failed. Please fix input data issues.")
        return False

    # ============ Step 1: Desurvey ============
    logger.info("=" * 50)
    logger.info("STEP 1: Desurvey")
    logger.info("=" * 50)
    from src import desurvey
    start = time.time()
    desurveyed = desurvey.run(
        data_dir=config.get('data_dir', 'data'),
        grade_field=grade_field
    )
    timings['desurvey_seconds'] = time.time() - start
    desurvey_path = os.path.join(output_dir, 'desurveyed.csv')
    desurveyed.to_csv(desurvey_path, index=False)

    # ============ Step 2: Compositing ============
    logger.info("=" * 50)
    logger.info("STEP 2: Compositing")
    logger.info("=" * 50)
    from src import composite
    start = time.time()
    composites = composite.run(
        data_dir=config.get('data_dir', 'data'),
        comp_length=config.get('compositing_length_m', 2.0),
        min_comp_length=config.get('min_composite_length_m', 0.5),
        grade_field=grade_field
    )
    timings['composite_seconds'] = time.time() - start
    composites_path = os.path.join(output_dir, 'composites.csv')
    composites.to_csv(composites_path, index=False)

    hard_domains_enabled = bool((config.get('domains', {}) or {}).get('hard_boundaries'))

    # ============ Step 3: Domains ============
    logger.info("=" * 50)
    logger.info("STEP 3: Domain Analysis")
    logger.info("=" * 50)
    from src import domains
    start = time.time()
    if hard_domains_enabled:
        domain_data = domains.build_categorical_domain_data(composites, config=config, grade_field=grade_field)
        domain_stats = {
            'categorical_counts': domain_data['domain_group'].value_counts().to_dict(),
            'lithology_counts': domain_data['lith_code'].value_counts().to_dict(),
        }
    else:
        domain_data, domain_stats = domains.run(
            composites_path=composites_path,
            target_lith_codes=config.get('target_lith_codes', ['GRSC']),
            grade_field=grade_field
        )
        top_cut_cfg = config.get('top_cut', {}) or {}
        if top_cut_cfg.get('enabled'):
            q = float(top_cut_cfg.get('quantile', 99.5))
            cap_val = float(domain_data[grade_field].quantile(q / 100.0))
            domain_data[grade_field] = domain_data[grade_field].clip(upper=cap_val)
            domain_stats = dict(domain_stats)
            domain_stats['top_cut'] = {
                'enabled': True,
                'quantile': q,
                'cap_value': cap_val,
                'n_capped': int((domain_data[grade_field] >= cap_val).sum()),
            }
    timings['domain_seconds'] = time.time() - start
    domain_path = os.path.join(output_dir, 'domain_data.csv')
    domain_data.to_csv(domain_path, index=False)
    declustered_path = None

    if hard_domains_enabled:
        from src import categorical_domains
        from src import cascade_grade

        logger.info("=" * 50)
        logger.info("STEP 4-7: Categorical Domains + Cascade Grade SGS")
        logger.info("=" * 50)
        start = time.time()
        cat_dir = os.path.join(output_dir, 'domains', 'categorical')
        categorical = categorical_domains.simulate_categorical_domains(
            composites=composites,
            config=config,
            output_dir=cat_dir,
        )
        grid_def = categorical['grid_def']
        prep = cascade_grade.prepare_domain_grade_models(
            composites=categorical['domain_df'],
            config=config,
            output_dir=output_dir,
            grade_field=grade_field,
        )
        sgs_result = cascade_grade.simulate_cascade_grades(
            domain_realizations=np.asarray(categorical['realizations']),
            models=prep['models'],
            grid_def=prep['grid_def'],
            config=config,
            output_dir=os.path.join(output_dir, 'grids'),
        )
        _save_sgs_outputs(
            sgs_result,
            config=config,
            output_dir=os.path.join(output_dir, 'grids'),
            n_real=int(config.get('simulation', {}).get('n_real', 100)),
            seed=int(config.get('simulation', {}).get('seed', 1337)),
        )
        domain_summary = {
            'categories': categorical['categories'],
            'cat_to_id': categorical['cat_to_id'],
            'probability_paths': categorical['probability_paths'],
            'realizations_path': categorical['realizations_path'],
            'state_path': categorical['state_path'],
        }
        with open(os.path.join(output_dir, 'tables', 'domain_assignment_summary.json'), 'w', encoding='utf-8') as f:
            json.dump(domain_summary, f, indent=2)

        first_model = prep['models']['fresh_graphitic'] if 'fresh_graphitic' in prep['models'] else next(iter(prep['models'].values()))
        primary_fig = os.path.join(first_model['figures_dir'], 'variogram.png')
        primary_json = os.path.join(first_model['figures_dir'], 'variogram_model.json')
        if os.path.exists(primary_fig):
            shutil.copy2(primary_fig, os.path.join(output_dir, 'figures', 'variogram.png'))
        if os.path.exists(primary_json):
            shutil.copy2(primary_json, os.path.join(output_dir, 'figures', 'variogram_model.json'))
        pair_counts = os.path.join(first_model['figures_dir'], 'variogram_pair_counts.csv')
        if os.path.exists(pair_counts):
            shutil.copy2(pair_counts, os.path.join(output_dir, 'figures', 'variogram_pair_counts.csv'))

        timings['decluster_seconds'] = 0.0
        timings['normal_score_seconds'] = 0.0
        timings['variography_seconds'] = 0.0
        timings['sgs_seconds'] = time.time() - start
        dc_stats = {
            'mode': 'categorical_domain_plus_cascade',
            'categorical_realizations': categorical['realizations_path'],
        }
        ranges = {
            name: {k: float(v) for k, v in model['ranges'].items()}
            for name, model in prep['models'].items()
        }
        nst_path = None
    else:
        # ============ Step 4: Declustering ============
        logger.info("=" * 50)
        logger.info("STEP 4: Declustering")
        logger.info("=" * 50)
        dc_config = config.get('declustering', {})
        from src import declustering
        start = time.time()
        declustered, dc_stats = declustering.run(
            data_path=domain_path,
            cell_size_xy=dc_config.get('cell_size_xy_m', 200),
            cell_size_z=dc_config.get('cell_size_z_m', 5),
            grade_field=grade_field
        )
        timings['decluster_seconds'] = time.time() - start
        declustered_path = os.path.join(output_dir, 'declustered.csv')
        declustered.to_csv(declustered_path, index=False)

        # ============ Step 5: Normal Score Transform ============
        logger.info("=" * 50)
        logger.info("STEP 5: Normal Score Transform")
        logger.info("=" * 50)
        from src import normal_score
        start = time.time()
        nst, nst_data = normal_score.run(
            data_path=declustered_path,
            grade_field=grade_field,
            config=config,
        )
        timings['normal_score_seconds'] = time.time() - start
        nst_path = os.path.join(output_dir, 'nst_data.csv')
        nst_data.to_csv(nst_path, index=False)

        # Save NST parameters
        nst_params_path = os.path.join(output_dir, 'nst_params.json')
        nst.save(nst_params_path)

        # ============ Step 6: Variography ============
        logger.info("=" * 50)
        logger.info("STEP 6: Variography")
        logger.info("=" * 50)
        from src import variography
        start = time.time()
        vario_model, exp_variograms, ranges = variography.run(
            data_path=nst_path,
            config=config,
            output_dir=os.path.join(output_dir, 'figures')
        )
        timings['variography_seconds'] = time.time() - start

        # ============ Step 7: SGS Simulation ============
        logger.info("=" * 50)
        logger.info("STEP 7: SGS Simulation")
        logger.info("=" * 50)
        from src import sgs
        start = time.time()
        sgs_result = sgs.run(
            data_path=nst_path,
            config=config,
            output_dir=os.path.join(output_dir, 'grids')
        )
        timings['sgs_seconds'] = time.time() - start

    # ============ Step 8: Risk Postprocessing ============
    logger.info("=" * 50)
    logger.info("STEP 8: Risk Postprocessing")
    logger.info("=" * 50)
    from src import postprocess_risk
    start = time.time()
    risk_results = postprocess_risk.run(
        config=config,
        output_dir=output_dir
    )
    timings['risk_postprocess_seconds'] = time.time() - start

    # ============ Step 9: Validation Plots ============
    logger.info("=" * 50)
    logger.info("STEP 9: Validation Plots")
    logger.info("=" * 50)
    from src import validation_plots
    start = time.time()
    plot_paths = validation_plots.run(
        data_path=domain_path,
        output_dir=output_dir,
        cutoff=config.get('cutoff_grade', 3.0),
        config=config,
        mode='reporting_support' if config.get('reporting_grid') else 'auto',
    )
    simulation_support_plots = None
    validation_cfg = config.get('validation', {}) or {}
    if validation_cfg.get('simulation_support_enabled'):
        simulation_support_plots = validation_plots.run(
            data_path=domain_path,
            output_dir=output_dir,
            cutoff=config.get('cutoff_grade', 3.0),
            config=config,
            mode='simulation_support',
            suffix='_2m',
            metrics_filename='validation_metrics_2m.json',
        )
    timings['validation_plots_seconds'] = time.time() - start

    # ============ Step 10: Drill Spacing Sensitivity ============
    sensitivity_cfg = config.get('sensitivity', {}) or {}
    if sensitivity_cfg.get('enabled', True):
        logger.info("=" * 50)
        logger.info("STEP 10: Drill Spacing Sensitivity")
        logger.info("=" * 50)
        from src import drill_spacing_sensitivity
        start = time.time()
        drill_spacing_sensitivity.run(config_path=config_path, output_dir=os.path.join(output_dir, 'sensitivity'))
        timings['sensitivity_seconds'] = time.time() - start
    else:
        logger.info("=" * 50)
        logger.info("STEP 10: Drill Spacing Sensitivity (skipped)")
        logger.info("=" * 50)
        timings['sensitivity_seconds'] = 0.0

    # ============ Step 11: Internal Validation (MODEL_OK vs SGS) ============
    logger.info("=" * 50)
    logger.info("STEP 11: Internal Validation")
    logger.info("=" * 50)
    internal_cfg = config.get('internal_validation', {}) or {}
    if internal_cfg.get('enabled', True):
        from src import internal_validation
        start = time.time()
        internal_val_status = internal_validation.run(config_path=config_path, output_dir=output_dir)
        timings['internal_validation_seconds'] = time.time() - start
        if internal_val_status.get('status') == 'failed':
            logger.error("Internal validation failed: %s", internal_val_status.get('error', 'unknown error'))
            return False
    else:
        internal_val_status = {'status': 'skipped', 'reason': 'disabled in config'}
        timings['internal_validation_seconds'] = 0.0

    # ============ Save Metadata ============
    logger.info("=" * 50)
    logger.info("Saving metadata")
    logger.info("=" * 50)

    metadata = {
        'config': config,
        'validation': validation,
        'domain_stats': domain_stats,
        'declustering': dc_stats,
        'variogram_ranges': ranges,
        'simulation': {
            'n_realizations': config.get('simulation', {}).get('n_real', 100),
            'seed': config.get('simulation', {}).get('seed', 1337)
        },
        'validation_outputs': {
            'primary': plot_paths,
            'simulation_support': simulation_support_plots,
        },
        'run_flags': {
            'runtime_profile': config.get('runtime_profile'),
            'calibration_enabled': bool(config.get('calibration', {}).get('enabled')),
            'calibration_method': config.get('calibration', {}).get('method'),
            'trend_enabled': bool(config.get('trend', {}).get('enabled')),
            'trend_columns': config.get('trend', {}).get('columns'),
            'neighborhood_enabled': True,
            'realizations_file': 'sgs_reals_calibrated.npy' if config.get('calibration', {}).get('enabled') else 'sgs_reals.npy',
            'internal_validation_enabled': bool(config.get('internal_validation', {}).get('enabled', True)),
            'internal_validation_status': internal_val_status,
        },
        'timings_seconds': timings,
        'output_files': {
            'desurveyed': desurvey_path,
            'composites': composites_path,
            'domain': domain_path,
            'declustered': declustered_path,
            'nst': nst_path,
            'sgs_reals': os.path.join(output_dir, 'grids', 'sgs_reals.npy'),
            'p10_grid': os.path.join(output_dir, 'grids', 'p10_grid.npy'),
            'p50_grid': os.path.join(output_dir, 'grids', 'p50_grid.npy'),
            'p90_grid': os.path.join(output_dir, 'grids', 'p90_grid.npy'),
            'risked_tonnage': os.path.join(output_dir, 'tables', 'risked_tonnage.csv')
        }
    }

    postrun_pack_cfg = config.get('postrun_review_pack', {}) or {}
    if postrun_pack_cfg.get('enabled', False):
        try:
            from src import reviewer_upgrade_pack

            metadata['postrun_review_pack'] = reviewer_upgrade_pack.run(
                output_dir=output_dir,
                config=config,
                top_n=int(postrun_pack_cfg.get('top_n', 50)),
            )
        except Exception as exc:
            logger.warning(f"Post-run reviewer pack failed: {exc}")
            metadata['postrun_review_pack'] = {
                'status': 'failed',
                'error': str(exc),
            }

    with open(os.path.join(output_dir, 'sgs_meta.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    # Update manuscript + tables
    paper_outputs_cfg = config.get('paper_outputs', {}) or {}
    if paper_outputs_cfg.get('auto_update_root_text', True):
        try:
            from src import paper_tables
            from src import update_manuscript
            tables_path = 'paper/tables.md' if os.path.exists('paper/tables.md') else 'tables.md'
            manuscript_path = 'paper/manuscript.md' if os.path.exists('paper/manuscript.md') else 'manuscript.md'
            paper_tables.run(output_dir=output_dir, tables_path=tables_path, config_path=config_path)
            update_manuscript.update(manuscript_path=manuscript_path, outputs_dir=output_dir, config_path=config_path)
        except Exception as exc:
            logger.warning(f"Post-run manuscript update skipped: {exc}")
    else:
        logger.info("Post-run root manuscript update disabled by config.")

    logger.info("=" * 50)
    logger.info("WORKFLOW COMPLETE!")
    logger.info("=" * 50)

    return True


def main():
    parser = argparse.ArgumentParser(description='Graphite SGS Workflow')
    parser.add_argument('--config', default='config/main_config.yaml', help='Config file path')
    parser.add_argument('--output', default='outputs', help='Output directory')
    parser.add_argument('--profile', default=None, help='Optional runtime profile from config/main_config.yaml (for example: fast2x or fast3x)')

    args = parser.parse_args()

    with _output_run_lock(args.output):
        success = run_full_workflow(args.config, args.output, profile_name=args.profile)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    import sys
    import os
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    main()