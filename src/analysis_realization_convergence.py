"""
Realization Convergence Test
============================
Tests whether 50 realizations is sufficient for stable SGS statistics.
Computes P10, P50, P90, mean, and variance at 10, 20, 30, 40, 50 realization subsets.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path

# Paths
OUTPUT_DIR = Path("output/a3_geology_aligned_250_200_20_nr100")
GRIDS_DIR = OUTPUT_DIR / "grids"
CHECKPOINT_FILE = GRIDS_DIR / "sgs_checkpoint_state.json"

def load_realizations(max_reals=None):
    """Load SGS realizations from checkpoint file."""
    # Load normal-score realizations
    reals_ns = np.load(GRIDS_DIR / "sgs_reals_ns_checkpoint.npy")
    
    # Load metadata
    with open(CHECKPOINT_FILE) as f:
        meta = json.load(f)
    
    n_completed = meta['completed_count'] + 1  # 0-indexed count
    shape = meta['shape']  # [realizations, z, y, x]
    
    print(f"Loaded {n_completed} completed realizations")
    print(f"Shape: {shape}")
    
    if max_reals is not None:
        n_use = min(max_reals, n_completed)
        reals_ns = reals_ns[:n_use]
        print(f"Using first {n_use} realizations")
    
    return reals_ns, meta

def compute_convergence_metrics(reals, percentiles=[10, 50, 90]):
    """Compute percentile maps and global statistics."""
    # Flatten spatial dimensions
    reals_flat = reals.reshape(reals.shape[0], -1)  # [n_real, n_cells]
    
    # Compute percentile maps
    p_maps = {}
    for p in percentiles:
        p_maps[p] = np.percentile(reals_flat, p, axis=0)
    
    # Global statistics
    stats = {
        'mean': np.mean(reals_flat, axis=0),
        'std': np.std(reals_flat, axis=0),
        'variance': np.var(reals_flat, axis=0),
    }
    
    return p_maps, stats, reals_flat

def convergence_analysis():
    """Main convergence analysis."""
    print("=" * 60)
    print("REALIZATION CONVERGENCE TEST")
    print("=" * 60)
    
    # Load all available realizations
    reals_ns, meta = load_realizations()
    n_total = reals_ns.shape[0]
    
    # Test at different realization counts
    test_counts = [10, 20, 30, 40, 50]
    test_counts = [n for n in test_counts if n <= n_total]
    
    results = {
        'n_real': [],
        'p10_mean': [], 'p10_std': [],
        'p50_mean': [], 'p50_std': [],
        'p90_mean': [], 'p90_std': [],
        'global_mean': [], 'global_mean_std': [],
        'global_var_mean': [], 'global_var_std': [],
    }
    
    print(f"\nTesting convergence at n = {test_counts}")
    print("-" * 60)
    
    for n in test_counts:
        subset = reals_ns[:n]
        p_maps, stats, flat = compute_convergence_metrics(subset)
        
        # Store results
        results['n_real'].append(n)
        results['p10_mean'].append(np.mean(p_maps[10]))
        results['p10_std'].append(np.std(p_maps[10]))
        results['p50_mean'].append(np.mean(p_maps[50]))
        results['p50_std'].append(np.std(p_maps[50]))
        results['p90_mean'].append(np.mean(p_maps[90]))
        results['p90_std'].append(np.std(p_maps[90]))
        results['global_mean'].append(np.mean(stats['mean']))
        results['global_mean_std'].append(np.std(stats['mean']))
        results['global_var_mean'].append(np.mean(stats['variance']))
        results['global_var_std'].append(np.std(stats['variance']))
        
        print(f"n={n:3d}: P10={np.mean(p_maps[10]):.4f}, P50={np.mean(p_maps[50]):.4f}, "
              f"P90={np.mean(p_maps[90]):.4f}, Mean={np.mean(stats['mean']):.4f}")
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Save results
    df.to_csv(OUTPUT_DIR / 'convergence_test_results.csv', index=False)
    print(f"\nResults saved to {OUTPUT_DIR / 'convergence_test_results.csv'}")
    
    # Create convergence plots
    create_convergence_figures(df)
    
    return df

def create_convergence_figures(df):
    """Create convergence diagnostic figures."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Percentile convergence with error bars
    ax1 = axes[0, 0]
    ax1.errorbar(df['n_real'], df['p10_mean'], yerr=df['p10_std'], 
                 fmt='o-', label='P10', capsize=3, color='blue')
    ax1.errorbar(df['n_real'], df['p50_mean'], yerr=df['p50_std'], 
                 fmt='s-', label='P50', capsize=3, color='green')
    ax1.errorbar(df['n_real'], df['p90_mean'], yerr=df['p90_std'], 
                 fmt='^-', label='P90', capsize=3, color='red')
    ax1.set_xlabel('Number of Realizations')
    ax1.set_ylabel('Mean Percentile Value (N-score)')
    ax1.set_title('Percentile Map Convergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Global mean convergence
    ax2 = axes[0, 1]
    ax2.errorbar(df['n_real'], df['global_mean'], yerr=df['global_mean_std'],
                 fmt='o-', color='purple', capsize=3)
    ax2.set_xlabel('Number of Realizations')
    ax2.set_ylabel('Global Mean (N-score)')
    ax2.set_title('Global Mean Convergence')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Variance convergence
    ax3 = axes[1, 0]
    ax3.errorbar(df['n_real'], df['global_var_mean'], yerr=df['global_var_std'],
                 fmt='o-', color='orange', capsize=3)
    ax3.set_xlabel('Number of Realizations')
    ax3.set_ylabel('Mean Cell Variance')
    ax3.set_title('Variance Convergence')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Coefficient of Variation of metrics
    ax4 = axes[1, 1]
    cv_p10 = df['p10_std'] / np.abs(df['p10_mean']) * 100
    cv_p50 = df['p50_std'] / np.abs(df['p50_mean']) * 100
    cv_p90 = df['p90_std'] / np.abs(df['p90_mean']) * 100
    ax4.plot(df['n_real'], cv_p10, 'o-', label='P10 CV%', color='blue')
    ax4.plot(df['n_real'], cv_p50, 's-', label='P50 CV%', color='green')
    ax4.plot(df['n_real'], cv_p90, '^-', label='P90 CV%', color='red')
    ax4.set_xlabel('Number of Realizations')
    ax4.set_ylabel('Coefficient of Variation (%)')
    ax4.set_title('Stability: Lower CV = More Stable')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, None)
    
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / 'figures/convergence_test.png', dpi=150, bbox_inches='tight')
    print(f"Figure saved to {OUTPUT_DIR / 'figures/convergence_test.png'}")
    
    # Print convergence summary
    print("\n" + "=" * 60)
    print("CONVERGENCE SUMMARY")
    print("=" * 60)
    
    # Calculate % change from n=50 to n=40, 30, 20, 10
    final_p50 = df[df['n_real'] == 50]['p50_mean'].values[0]
    final_p10 = df[df['n_real'] == 50]['p10_mean'].values[0]
    final_p90 = df[df['n_real'] == 50]['p90_mean'].values[0]
    
    print(f"\nReference (n=50): P10={final_p10:.4f}, P50={final_p50:.4f}, P90={final_p90:.4f}")
    print("\n% Difference from n=50:")
    
    for n in [40, 30, 20, 10]:
        row = df[df['n_real'] == n]
        if len(row) > 0:
            p50_diff = (row['p50_mean'].values[0] - final_p50) / final_p50 * 100
            p10_diff = (row['p10_mean'
