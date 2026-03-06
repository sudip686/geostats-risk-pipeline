import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from graphite_sgs.utils import load_data, calculate_tonnage_curve
from graphite_sgs.desurvey import process_holes
from graphite_sgs.compositing import composite_drills
from graphite_sgs.sgs import run_sgs
from graphite_sgs.kriging import run_ok
from graphite_sgs.analysis import run_classification, plot_sensitivity_curves, plot_smoothing_comparison

def main():
    print("=== Graphite SGS Advanced Sensitivity Study ===")
    
    # Configuration
    DATA_DIR = 'data'
    OUTPUT_DIR = 'outputs/study'
    DOMAIN = ['GRSC', 'GRSC1', 'GRSC2', 'GRSC3']
    DX, DY, DZ = 100.0, 100.0, 10.0 # Very coarse for demo speed
    NREAL = 3 # Minimal for valid execution
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # 1. Load Data
    print("Loading data...")
    collar, survey, assay, litho = load_data(DATA_DIR)
    xyz_data = process_holes(collar, survey, assay, litho)
    composites = composite_drills(xyz_data, comp_length=5.0)
    
    # Filter Domain
    data = composites[composites['lith_code'].isin(DOMAIN)].copy()
    print(f"Total Composites: {len(data)}")
    
    # Define Grid
    min_x, max_x = data['x'].min() - 50, data['x'].max() + 50
    min_y, max_y = data['y'].min() - 50, data['y'].max() + 50
    min_z, max_z = data['z'].min() - 10, data['z'].max() + 10
    
    # Snap
    min_x = np.floor(min_x / DX) * DX
    min_y = np.floor(min_y / DY) * DY
    min_z = np.floor(min_z / DZ) * DZ
    
    grid_def = {
        'x': [min_x, max_x, DX],
        'y': [min_y, max_y, DY],
        'z': [min_z, max_z, DZ]
    }
    
    # 2. Scenario A: Base Case (All Data)
    print("\n--- Running Base Case (All Data) ---")
    
    # SGS Base
    print("Running SGS (Base)...")
    sgs_base = run_sgs(data, DOMAIN, grid_def, n_realizations=NREAL)
    
    # OK Base
    print("Running OK (Base)...")
    ok_base = run_ok(data, DOMAIN, grid_def)
    
    # 3. Scenario B: Sparse Data (Sensitivity)
    print("\n--- Running Sensitivity Case (Sparse Data) ---")
    
    # Identify holes to keep
    # Randomly select 60% of holes
    unique_holes = data['hole_id'].unique()
    np.random.seed(42)
    keep_holes = np.random.choice(unique_holes, size=int(len(unique_holes) * 0.6), replace=False)
    
    data_sparse = data[data['hole_id'].isin(keep_holes)].copy()
    print(f"Sparse Composites: {len(data_sparse)} (from {len(unique_holes)} holes to {len(keep_holes)})")
    
    # SGS Sparse
    print("Running SGS (Sparse)...")
    sgs_sparse = run_sgs(data_sparse, DOMAIN, grid_def, n_realizations=NREAL)
    
    # 4. Analysis & Comparison
    print("\n--- Generating Analysis ---")
    
    # A. Smoothing Effect (OK vs SGS Base)
    plot_smoothing_comparison(
        ok_base['est'], 
        sgs_base['realizations'], 
        os.path.join(OUTPUT_DIR, 'smoothing_comparison.png')
    )
    
    # B. Uncertainty / Sensitivity
    # Calculate curves
    cutoffs = np.linspace(0, 15, 16)
    vol = DX * DY * DZ
    
    curves_base = calculate_tonnage_curve(sgs_base['realizations'], cutoffs, vol)
    curves_sparse = calculate_tonnage_curve(sgs_sparse['realizations'], cutoffs, vol)
    
    plot_sensitivity_curves(
        curves_base, 
        curves_sparse, 
        cutoffs, 
        os.path.join(OUTPUT_DIR, 'drill_spacing_sensitivity.png')
    )
    
    # Quantitative Uncertainty Reduction
    # Compare width of P90-P10 intervals at 5% cutoff
    base_p10_tonnes = curves_base.loc[curves_base['cutoff'] == 5.0, 'tonnage_p10'].values[0]
    base_p90_tonnes = curves_base.loc[curves_base['cutoff'] == 5.0, 'tonnage_p90'].values[0]
    base_uncert = (base_p10_tonnes - base_p90_tonnes) # Tonnage decreases with cutoff? No, curve is T > cut. P10 is low tonnage (conservative)?
    # P10 Tonnage: 10% chance tonnage is LESS than this? No.
    # Usually: P90 is High Confidence (Conservative, Low Tonnage). P10 is Low Confidence (Optimistic, High Tonnage).
    # Check np.percentile logic in utils.
    # 'tonnage_p10': np.percentile(tonnages, 10). Tonnages is list of T across reals.
    # P10 = small number. P90 = large number.
    # So Uncertainty Range = P90 - P10.
    
    sparse_p10_tonnes = curves_sparse.loc[curves_sparse['cutoff'] == 5.0, 'tonnage_p10'].values[0]
    sparse_p90_tonnes = curves_sparse.loc[curves_sparse['cutoff'] == 5.0, 'tonnage_p90'].values[0]
    
    base_width = base_p90_tonnes - base_p10_tonnes
    sparse_width = sparse_p90_tonnes - sparse_p10_tonnes
    
    reduction = (sparse_width - base_width) / sparse_width * 100
    
    with open(os.path.join(OUTPUT_DIR, 'sensitivity_report.txt'), 'w') as f:
        f.write("Drill Spacing Sensitivity Report\n")
        f.write("================================\n")
        f.write(f"Base Hole Count: {len(unique_holes)}\n")
        f.write(f"Sparse Hole Count: {len(keep_holes)}\n\n")
        f.write(f"Uncertainty Width (Tonnes @ 5% Cutoff):\n")
        f.write(f"  Sparse: {sparse_width:,.0f} t\n")
        f.write(f"  Base:   {base_width:,.0f} t\n")
        f.write(f"  Reduction: {reduction:.1f}%\n")
    
    # C. Classification Grid (Base Case)
    print("Running Classification...")
    run_classification(
        ok_base['est'], 
        ok_base['var'], 
        sgs_base['realizations'], 
        grid_def, 
        OUTPUT_DIR
    )
    
    # D. Economic Sensitivity Tables
    print("Generating Economic Tables...")
    econ_cutoffs = [3.0, 5.0, 7.0]
    econ_report = []
    
    for cut in econ_cutoffs:
        row = curves_base[curves_base['cutoff'] == cut].iloc[0]
        econ_report.append({
            'Cutoff (%)': cut,
            'Tonnes (P90) [Conservative]': row['tonnage_p10'], # Assuming P10 is small tonnage (Conservative? No, standard is P90=Conservative in Geostats? Wait.)
            # Standard:
            # P90: 90% probability tonnage is GREATER than this. (Low value) -> Conservative.
            # P10: 10% probability tonnage is GREATER than this. (High value) -> Optimistic.
            # numpy percentile 10 gives low value. 
            # If Distribution is of "Available Tonnage", then 10th percentile is small tonnage.
            # Probability T > X = 0.9 means X is small.
            # So P10 (numpy) is the value where 10% of data is below.
            # So P10 is the Conservative estimate (90% chance it's higher).
            # Wait. P10 (numpy) = Value x. 10% of sims < x. 90% > x.
            # So P10 is the "Proven" / Conservative number.
            # P90 (numpy) = Value y. 90% of sims < y. 10% > y. Optimistic.
            
            # Let's label explicitly based on numpy stats.
            'Tonnes_P10 (Conservative)': row['tonnage_p10'],
            'Tonnes_P50 (Median)': row['tonnage_p50'],
            'Tonnes_P90 (Optimistic)': row['tonnage_p90'],
            'Grade_P50': row['grade_p50']
        })
        
    pd.DataFrame(econ_report).to_csv(os.path.join(OUTPUT_DIR, 'economic_sensitivity.csv'), index=False)
    
    print("Study Complete. Check outputs/study/")

if __name__ == "__main__":
    main()
