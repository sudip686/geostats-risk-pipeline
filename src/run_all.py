"""
run_all.py - Main CLI Entrypoint

Runs the complete SGS workflow from configuration.
"""

import argparse
import logging
import os
import sys
import json
import time
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def run_full_workflow(config_path='config/project.yaml', output_dir='outputs'):
    """
    Run the complete SGS workflow.

    Args:
        config_path: Path to YAML configuration
        output_dir: Output directory
    """
    from src.utils.io import load_config

    # Load config
    logger.info(f"Loading configuration from {config_path}")
    config = load_config(config_path)

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
    validation = validate_inputs.run_validation(data_dir=config.get('data_dir', 'data'))
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

    # ============ Step 3: Domains ============
    logger.info("=" * 50)
    logger.info("STEP 3: Domain Analysis")
    logger.info("=" * 50)
    from src import domains
    start = time.time()
    domain_data, domain_stats = domains.run(
        composites_path=composites_path,
        target_lith_codes=config.get('target_lith_codes', ['GRSC']),
        grade_field=grade_field
    )
    timings['domain_seconds'] = time.time() - start
    domain_path = os.path.join(output_dir, 'domain_data.csv')
    domain_data.to_csv(domain_path, index=False)

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
    if os.environ.get('CI', '').lower() in {'1', 'true', 'yes'}:
        config = dict(config)
        sim = dict(config.get('simulation', {}))
        sim['n_real'] = config.get('ci', {}).get('n_real', sim.get('n_real', 100))
        config['simulation'] = sim

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
        data_dir=config.get('data_dir', 'data'),
        output_dir=output_dir,
        cutoff=config.get('cutoff_grade', 3.0),
        config=config,
    )
    timings['validation_plots_seconds'] = time.time() - start

    # ============ Step 10: Drill Spacing Sensitivity ============
    logger.info("=" * 50)
    logger.info("STEP 10: Drill Spacing Sensitivity")
    logger.info("=" * 50)
    from src import drill_spacing_sensitivity
    start = time.time()
    drill_spacing_sensitivity.run(config_path=config_path, output_dir=os.path.join(output_dir, 'sensitivity'))
    timings['sensitivity_seconds'] = time.time() - start

    # ============ Step 11: Internal Validation (MODEL_OK vs SGS) ============
    logger.info("=" * 50)
    logger.info("STEP 11: Internal Validation")
    logger.info("=" * 50)
    from src import internal_validation
    start = time.time()
    internal_val_status = internal_validation.run(config_path=config_path, output_dir=output_dir)
    timings['internal_validation_seconds'] = time.time() - start

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
        'run_flags': {
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

    with open(os.path.join(output_dir, 'sgs_meta.json'), 'w') as f:
        json.dump(metadata, f, indent=2, default=_json_default)

    # Update manuscript + tables
    try:
        from src import paper_tables
        from src import update_manuscript
        paper_tables.run(output_dir=output_dir, tables_path='paper/tables.md', config_path=config_path)
        update_manuscript.update(manuscript_path='paper/manuscript.md', outputs_dir=output_dir, config_path=config_path)
    except Exception as exc:
        logger.warning(f"Post-run manuscript update skipped: {exc}")

    logger.info("=" * 50)
    logger.info("WORKFLOW COMPLETE!")
    logger.info("=" * 50)

    return True


def main():
    parser = argparse.ArgumentParser(description='Graphite SGS Workflow')
    parser.add_argument('--config', default='config/project.yaml', help='Config file path')
    parser.add_argument('--output', default='outputs', help='Output directory')

    args = parser.parse_args()

    success = run_full_workflow(args.config, args.output)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    import sys
    import os
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    main()
