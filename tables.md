# Tables for Graphite SGS Paper

## Table 1: GRSC Statistics from Current Run (`outputs/declustered.csv`, lith_code = GRSC)

| Statistic | Value | Units |
|-----------|-------|-------|
| Number of Composites | 1,790 | count |
| Mean TGC | 3.97 | % |
| Standard Deviation | 2.24 | % |
| Variance | 5.02 | %^2 |
| Coefficient of Variation | 0.56 | dimensionless |
| Minimum | 0.05 | % |
| Maximum | 14.67 | % |
| Median | 3.51 | % |
| 25th Percentile | 2.31 | % |
| 75th Percentile | 5.25 | % |

## Table 2: Declustering Summary (Combined Simulation Domain)

| Parameter | Value | Units |
|-----------|-------|-------|
| Method | Cell Declustering | - |
| Cell Size X | 200 | m |
| Cell Size Y | 200 | m |
| Cell Size Z | 5 | m |
| Raw Mean | 4.37 | % TGC |
| Declustered Mean | 4.06 | % TGC |
| Declustering Ratio | 0.93 | dimensionless |
| Minimum Weight | 0.12 | dimensionless |
| Maximum Weight | 3.36 | dimensionless |
| Average Weight | 1.00 | dimensionless |

## Table 3: Variogram Parameters (Normal-Score Domain)

| Parameter | Along Strike | Down Dip | Normal to Plane | Units |
|-----------|--------------|----------|-----------------|-------|
| Azimuth | 105 | 15 | 195 | degrees |
| Dip | 0 | 32 | 58 | degrees |
| Nugget (C0) | 0.24 | 0.24 | 0.24 | dimensionless |
| Structured Sill (C) | 0.96 | 0.96 | 0.96 | dimensionless |
| Total Sill (C0 + C) | 1.20 | 1.20 | 1.20 | dimensionless |
| Range (a) | 237.5 | 188.5 | 112.5 | m |
| Nugget Ratio | 0.2 | 0.2 | 0.2 | dimensionless |
| Model Type | Exponential | Exponential | Exponential | - |
## Table 4: Grid Definition

| Parameter | Value | Units |
|-----------|-------|-------|
| Coordinate System | UTM Zone 37S | - |
| EPSG Code | 32737 | - |
| Origin X | 475,500 | m |
| Origin Y | 9,465,400 | m |
| Origin Z | 570 | m |
| Number of Cells (NX) | 13 | - |
| Number of Cells (NY) | 46 | - |
| Number of Cells (NZ) | 27 | - |
| Cell Size (DX) | 100 | m |
| Cell Size (DY) | 100 | m |
| Cell Size (DZ) | 10 | m |
| Total Cells | 16,146 | - |
| X Extent | 1,200 | m |
| Y Extent | 4,500 | m |
| Z Extent | 260 | m |

## Table 5: Simulation Parameters

| Parameter | Value | Units |
|-----------|-------|-------|
| Number of Realizations | 200 | count |
| Random Seed | 1337 | - |
| Kriging Type | Ordinary Kriging | - |
| Search Neighborhood | Anisotropic local ellipsoid (250 x 120 x 70 m) | - |
| Minimum Neighbors | 8 | count |
| Maximum Neighbors | 24 | count |
| Anisotropy (Major/Minor) | 3.60 | dimensionless |
| Anisotropy (Major/Intermediate) | 2.25 | dimensionless |
## Table 6: Methodological Pilot-Screen Scenario Tonnage (Non-Resource)

| Cutoff (% TGC) | P10 Tonnage (Mt) | P50 Tonnage (Mt) | P90 Tonnage (Mt) | P50 Grade (% TGC) | P50 Contained (kt) |
|---|---|---|---|---|---|
| 0 | 215.79 | 215.79 | 215.79 | 4.86 | 10488 |
| 1 | 215.79 | 215.79 | 215.79 | 4.86 | 10488 |
| 2 | 215.59 | 215.68 | 215.74 | 4.86 | 10486 |
| 3 | 209.52 | 210.53 | 211.18 | 4.91 | 10342 |
| 4 | 162.20 | 165.39 | 168.57 | 5.28 | 8722 |
| 5 | 84.14 | 87.81 | 91.25 | 5.96 | 5230 |
| 6 | 30.00 | 32.46 | 34.57 | 6.85 | 2222 |
| 7 | 8.73 | 10.02 | 11.14 | 7.74 | 777 |
| 8 | 2.30 | 2.78 | 3.26 | 8.66 | 241 |
| 9 | 0.29 | 0.44 | 0.60 | 9.61 | 42 |
| 10 | 0.05 | 0.11 | 0.17 | 10.54 | 11 |
## Table 7: Internal Validation Metrics (Baseline vs Final)

| Metric | Baseline | Final | Source |
|--------|----------|-------|--------|
| Mean sim grade (%) | 3.9218 | 4.8532 | `validation_metrics_pre.json` / `validation_metrics.json` |
| Sim std (%) | 2.3014 | 1.1513 | `validation_metrics_pre.json` / `validation_metrics.json` |
| Histogram overlap | 0.5599 | 0.9789 | `validation_metrics_pre.json` / `validation_metrics.json` |
| Q-Q RMSE | 1.4987 | 0.0760 | `validation_metrics_pre.json` / `validation_metrics.json` |
| Swath corr X | 0.6308 | 0.5844 | `validation_metrics_pre.json` / `validation_metrics.json` |
| Swath corr Y | 0.5585 | 0.6229 | `validation_metrics_pre.json` / `validation_metrics.json` |
| Swath corr Z | 0.4038 | 0.4123 | `validation_metrics_pre.json` / `validation_metrics.json` |
| Swath coverage (P10-P90, %) | 100.00 | 87.33 | `validation_metrics_pre.json` / `validation_metrics.json` |

## Table 8: Interim Random-Fold Cross-Validation Check (Internal, Non-Spatial)

| CV Scenario | Source file | Samples (n) | Folds | ME | MAE | RMSE | RMSE / Std(data) |
|-------------|-------------|-------------|-------|----|-----|------|------------------|
| Blocked CV (primary) | `cross_validation_blocked_300.json` | 300 | 5 | -0.1859 | 2.0674 | 2.6153 | 2.28 |
| Base CV | `cross_validation_300.json` | 300 | 5 | -0.0661 | 1.7358 | 2.2823 | 1.99 |
| Robustness CV | `cross_validation_600.json` | 600 | 5 | 0.0408 | 1.5430 | 2.1272 | 1.85 |

Note: Blocked CV uses XY spatial blocks (500 m) and is treated as primary predictive evidence; random-fold rows are secondary diagnostics.

## Table 9: Variogram Pair Counts by Lag

| Direction | Lag | Pair Count |
|-----------|-----|------------|
| along_strike | 1 | 274 |
| along_strike | 2 | 824 |
| along_strike | 3 | 1524 |
| along_strike | 4 | 2436 |
| along_strike | 5 | 2812 |
| along_strike | 6 | 1152 |
| along_strike | 7 | 1448 |
| along_strike | 8 | 2796 |
| along_strike | 9 | 1736 |
| along_strike | 10 | 834 |
| down_dip | 1 | 200 |
| down_dip | 2 | 548 |
| down_dip | 3 | 1336 |
| down_dip | 4 | 1050 |
| down_dip | 5 | 784 |
| down_dip | 6 | 818 |
| down_dip | 7 | 1386 |
| down_dip | 8 | 300 |
| down_dip | 9 | 458 |
| down_dip | 10 | 62 |
| normal_to_plane | 1 | 26 |
| normal_to_plane | 2 | 160 |
| normal_to_plane | 3 | 174 |
| normal_to_plane | 4 | 10 |
| normal_to_plane | 5 | 6 |
| normal_to_plane | 6 | 0 |
| normal_to_plane | 7 | 0 |
| normal_to_plane | 8 | 0 |
| normal_to_plane | 9 | 0 |
| normal_to_plane | 10 | 0 |

Note: For directional model reliability, at least ~30 pairs per lag is a practical minimum rule-of-thumb. The normal-to-plane direction does not meet this at most lags.

## Table 10: Trend Diagnostics (Declustered Domain)

| Metric | Value |
|--------|-------|
| Samples (n) | 3506 |
| Trend intercept (beta0) | 0.0455633 |
| Trend slope in z (beta1) | 0.00602013 |
| Raw grade-vs-z slope | 0.00602013 |
| Residual slope after detrending | 1.04e-18 |
| Trend R^2 | 0.0117 |

## Table 11: Compositing Support Sensitivity (Combined Simulation Domain)

| Composite length (m) | Composites (n) | Mean TGC (%) | Std TGC (%) | Median TGC (%) |
|----------------------|----------------|--------------|-------------|----------------|
| 1.0 | 6815 | 4.3752 | 2.3477 | 4.04 |
| 2.0 | 3506 | 4.3701 | 2.3017 | 4.04 |
| 3.0 | 2395 | 4.3752 | 2.2254 | 4.07 |

## Table 12: Trend/Stationarity Reconciliation

| Metric | Value | Source |
|--------|-------|--------|
| Trend model slope in z | 0.00602013 | `trend_diagnostics.json` |
| Stationarity check slope in z | 0.00578190 | `stationarity_trends.json` |
| Difference | 0.00023823 | computed |
| Reconciliation note | Different fitting datasets/targets | `trend_stationarity_reconciliation.json` |

## Table 13: Reproducibility Manifest for Reported Outputs

| Item | Primary source file(s) |
|------|------------------------|
| Table 6 risk screening (scenario-scaled) | `risked_tonnage.csv` |
| Table 7 internal validation diagnostics | `validation_metrics_pre.json`; `validation_metrics.json` |
| Table 8 cross-validation | `cross_validation_300.json`; `cross_validation_600.json` |
| Table 8 blocked cross-validation | `cross_validation_blocked_300.json` |
| Table 9 pair counts | `variogram_pair_counts.csv` |
| Table 10 trend diagnostics | `trend_diagnostics.json` |
| Table 12 trend/stationarity reconciliation | `stationarity_trends.json`; `trend_stationarity_reconciliation.json` |
| Table 14 internal continuity check | `validation_metrics_pre.json`; `validation_metrics.json` |
| Table 15 normal-range sensitivity | `outputs/tables/normal_range_sensitivity.csv` |
| Table 16 variogram reproduction check | `outputs/tables/variogram_reproduction_summary.json`; `outputs/tables/variogram_reproduction_lag.csv` |
| Figure 1 variogram | `outputs/figures/variogram.png`; `outputs/figures/variogram_model.json` |
| Figures 2-4 validation plots | `outputs/figures/histogram_validation.png`; `outputs/figures/qq_plot.png`; `outputs/figures/swath_x.png`; `outputs/figures/swath_y.png`; `outputs/figures/swath_z.png` |
| Figure 5 trend plot | `outputs/figures/trend_diagnostic.png` |
| Figure 6 support histogram | `outputs/figures/composite_length_hist.png` |

## Table 14: Internal Continuity Check (Baseline vs Final)

| Metric | Baseline | Final | Delta (Final-Baseline) | Source |
|--------|----------|-------|------------------------|--------|
| Swath corr X | 0.6308 | 0.5844 | -0.0464 | `validation_metrics_pre.json`; `validation_metrics.json` |
| Swath corr Y | 0.5585 | 0.6229 | +0.0644 | `validation_metrics_pre.json`; `validation_metrics.json` |
| Swath corr Z | 0.4038 | 0.4123 | +0.0085 | `validation_metrics_pre.json`; `validation_metrics.json` |

Note: This table is a minimal internal continuity check to show swath-correlation changes are limited in magnitude.

## Table 15: Normal-Range Sensitivity (Configured Anisotropy)

| Scenario | Configured normal range (m) | Major/normal anisotropy ratio | Z-swath corr | Mean cell P(grade >3%) | P50 tonnage at 3% cutoff (Mt) | Source |
|----------|------------------------------|--------------------------------|--------------|-------------------------|-------------------------------|--------|
| Best-fit base | 70 | 5.1 | 0.4123 | 0.9752 | 210.53 | `outputs/tables/validation_metrics.json`; `outputs/tables/risked_tonnage.csv`; `outputs/grids/prob_gt_3.0.npy` |
| Sensitivity A | 60 | 6.0 | 0.3878 | 0.9752 | 210.48 | `outputs/tables/normal_range_sensitivity.csv` |
| Sensitivity B | 150 | 2.4 | 0.4085 | 0.9752 | 210.51 | `outputs/tables/normal_range_sensitivity.csv` |

Note: Normal-direction pair support is weak (Table 9), so this table is used to bound interpretation sensitivity rather than claim robust normal-direction continuity.

## Table 16: Variogram Reproduction Check (Target vs Realization Ensemble, Grade Space)

| Direction | Valid lags used | Gamma RMSE | Gamma MAE | Source |
|-----------|-----------------|------------|-----------|--------|
| along_strike | 8 | 1.109 | 0.889 | `outputs/tables/variogram_reproduction_summary.json` |
| down_dip | 9 | 0.839 | 0.699 | `outputs/tables/variogram_reproduction_summary.json` |

Note: This check is a pilot-support grade-space proxy; shortest lag behavior is constrained by 100 m block support and reduced short-distance pair availability in the simulated grid (`variogram_reproduction_lag.csv`).
