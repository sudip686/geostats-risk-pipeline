# Grade Uncertainty and Risk Assessment in a Stratiform Graphite Deposit, Tanzania: A Sequential Gaussian Simulation Approach

**Authors:** Blinded for review

**Affiliations:** Blinded for review

**Corresponding Author:** Blinded for review

---

## Abstract

This study presents a geostatistical assessment of grade uncertainty in a stratiform graphite deposit in Tanzania. Sequential Gaussian Simulation (SGS) generated 200 conditional realizations of total graphitic carbon (TGC) on a pilot grid covering about 1.2 km x 4.5 km (about 5.4 km2) with 100 m x 100 m x 10 m blocks. The workflow uses minimum-curvature desurveying, 2 m length-weighted compositing, cell declustering (200 m x 200 m x 5 m), weighted normal-score transformation, directional variography, SGS, trend add-back, and post-simulation quantile-mapping calibration.

Directional continuity is anisotropic with exponential model ranges of 237.5 m (along strike), 188.5 m (down dip), and 112.5 m (normal to plane), nugget 0.24 and structured sill 0.96 in normal-score space. Normal-to-plane continuity is poorly constrained by pair support and is treated as provisional; sensitivity tests on configured normal range are reported. Under the explicit pilot-screen scenario factor ($f_v = 0.055$), at 3% TGC cutoff the calibrated SGS ensemble gives P10/P50/P90 = 209.52/210.53/211.18 Mt with P50 grade 4.91% TGC.

These outputs are for pilot-scale uncertainty screening and methodological evaluation only. They do not constitute a Mineral Resource or Ore Reserve statement under any reporting code.

**Keywords:** graphite; sequential Gaussian simulation; geostatistics; uncertainty; risk; Tanzania

---

## 1. Introduction

### 1.1 Problem and decision context

Deterministic kriging smooths local variability and can understate uncertainty. SGS is used here to represent uncertainty while honoring conditioning data and spatial continuity (Journel and Huijbregts, 1978; Deutsch and Journel, 1998; Goovaerts, 1997; Chiles and Delfiner, 2012; Nowak and Verly, 2005; Zanon and Leuangthong, 2003).

### 1.2 Study positioning and precedent

This paper documents a reproducible pilot-scale SGS workflow and reports uncertainty products (percentiles, exceedance probability, and risked tonnage) for a stratiform graphite setting.

A directly relevant African mining precedent reports grade-tonnage uncertainty using stochastic geostatistical workflows that combine multiple-point geostatistics with SGS in copper mineralization (Paithankar and Chatterjee, 2018). This provides regional precedent for framing pilot-scale uncertainty outputs as decision-support products rather than deterministic resource-style claims.

Scope statement: this manuscript reports pilot-scale uncertainty screening outputs only. It is not a public resource statement and must not be used as JORC or NI 43-101 compliant reporting.

## 2. Geological Context

Mineralization is hosted in graphitic schist with stratiform geometry and dominant lateral continuity. Model orientation used in this study is strike 105 degrees, dip 32 degrees, dip direction 15 degrees (dip positive down). The modeled weathering profile (oxide, transition, fresh) is used for density weighting in risk postprocessing.

## 3. Data and Methods

### 3.1 Drillhole database and validation

Validated inputs include collars, surveys, assays, and lithology. Summary counts from run metadata:

- Drillholes: 104
- Survey records: 208
- Assay intervals: 3336
- Lithology intervals: 1248
- Total validated assay meters: 7872.27 m

Validation checks covered interval integrity, depth monotonicity, survey ranges, and assay ranges.

### 3.2 Desurveying

Assay intervals were desurveyed using minimum curvature. Sample coordinates were assigned at interval midpoints along the desurveyed trajectory.

### 3.3 Compositing and support

Assays were composited to 2.0 m support using length weighting:

$$Z_{comp} = \frac{\sum_i L_i Z_i}{\sum_i L_i}$$

Minimum composite length was 0.5 m. Support sensitivity (1 m, 2 m, 3 m) is summarized in Table 11 and shows stable domain means with increasing smoothing (lower standard deviation at longer support).

### 3.4 Domain definition

The simulation domain includes lithology codes: GRSC, GRSC1, GRSC2, SAP (GRSC), and SAPR (GRSC). Combined domain composite count in run metadata is 3506.

GRSC-only statistics from the current run are reported in Table 1 (n = 1790; mean = 3.97% TGC; std = 2.24%). Table 2 reports weighted combined-domain declustering statistics; these tables are different populations and are not numerically interchangeable.

### 3.5 Declustering

Cell declustering used 200 m x 200 m x 5 m cells with weights:

$$w_i = \frac{1}{n_{cell(i)}}$$

Declustering summary:

- Raw mean: 4.37% TGC
- Declustered mean: 4.06% TGC
- Weight range: 0.12 to 3.36 (mean 1.00)

### 3.6 Normal-score transform

Normal-score transform was fitted on declustered grades using centered weighted cumulative probabilities:

$$p_i = \frac{\sum_{j\le i} w_j - 0.5 w_i}{\sum_j w_j}$$

Normal scores were obtained as $y_i = \Phi^{-1}(p_i)$. Decluster weights were used when available; otherwise unit weights were used. Back-transform used piecewise linear interpolation between fitted score-data pairs with score clipping to fitted score bounds.

### 3.7 Trend handling

A linear trend in z was fitted by least squares and removed before NST/SGS, then added back after back-transform:

$$T(z) = \beta_0 + \beta_1 z$$

with fitted coefficients from run metadata: $\beta_0 = 0.0455633$, $\beta_1 = 0.00602013$. Trend diagnostics are summarized in Table 10, and reconciliation against `stationarity_trends.json` is given in Table 12.

### 3.8 Internal validation protocol

Validation is performed using internal project data and realization diagnostics only. Four checks are used:

1. Distribution reproduction: histogram overlap and Q-Q RMSE between model outputs and internal reference distributions.
2. Spatial reproduction: swath diagnostics in X, Y, and Z using mean trends and P10-P90 envelopes.
3. Predictive diagnostics: random-fold and blocked spatial cross-validation, with blocked CV treated as primary predictive evidence.
4. Structural diagnostics: lag-wise variogram reproduction checks against target directional continuity.

These checks are complementary. Distribution and swath checks evaluate ensemble realism at support scale, while cross-validation quantifies point-support predictive difficulty.

### 3.9 Coordinate reference system

All modeling was performed in EPSG:32737 (UTM Zone 37S).

### 3.10 Variography

Directional variograms were computed in normal-score space using:

- Along strike: azimuth 105 degrees, dip 0 degrees
- Down dip: azimuth 15 degrees, dip 32 degrees
- Normal to plane: azimuth 195 degrees, dip 58 degrees

Variogram setup:

- Lag distance: 50 m
- Number of lags: 10
- Maximum distance: 500 m
- Angular tolerance: 22.5 degrees
- Fit method: weighted least squares
- Candidate models: exponential, spherical
- Variogram subsample cap: 800 composites
- Pair cap per direction: 200000
- Lag binning: equal-width bins from 0 to max distance
- Variogram fitting used valid semivariances only (finite, >0)
- Fitted range was constrained to practical bounds (invalid/extreme fitted values reset to half max lag)

Selected model for simulation: exponential with nugget 0.24, structured sill 0.96, and ranges 237.5 m (strike), 188.5 m (down dip), 112.5 m (normal).

Lag-wise pair counts are reported in Table 9; normal-to-plane support is sparse at higher lags (zero pairs from lag 6 onward).
Normal-to-plane continuity is poorly constrained; it is treated as provisional and tested with normal-range sensitivity (Table 15).
Accordingly, continuity interpretation is treated as robust in strike/down-dip directions and provisional in the normal direction.
This treatment is consistent with screening-stage guidance on variogram reproduction and neighborhood/search-strategy interactions in sequential simulation workflows (Babak, 2006).

### 3.11 SGS setup

Pilot grid and simulation parameters:

- Origin: (475500, 9465400, 570)
- Grid size: 13 x 46 x 27 cells
- Cell size: 100 m x 100 m x 10 m
- Total cells: 16146
- Realizations: 200
- Seed: 1337
- Kriging: ordinary kriging
- Search ellipsoid radii: 250 m (strike), 120 m (down dip), 70 m (normal)
- Neighbors: minimum 8, maximum 24

Simulation used local sequential conditioning with anisometric coordinates. If minimum neighbors were not found within the initial search radius, the radius was expanded by a factor of 1.5 up to 4 times; if still insufficient, nearest-neighbor fallback was applied to satisfy the minimum-neighbor constraint.

### 3.12 Uncertainty products

From 200 realizations, the study reports per-cell P10, P50, P90 and exceedance probabilities:

$$P(Z(u) > c) = \frac{1}{R}\sum_{r=1}^{R} I\left(Z^{(r)}(u) > c\right)$$

Primary probability map cutoff is c = 3% TGC.

### 3.13 Risked tonnage formulation

For each cutoff c and realization r:

$$T_{r,\mathrm{gross}}(c) = \sum_u I\left(Z^{(r)}(u) \ge c\right) \cdot V_{block} \cdot \rho$$

$$T_{r,\mathrm{scaled}}(c) = T_{r,\mathrm{gross}}(c)\cdot f_v$$

with:

- $V_{block}$ = 100 x 100 x 10 = 100000 m3
- $\rho$ = 2.43 t/m3 (weathering-weighted basis)
- $f_v$ = 0.055 (scenario scaling factor for effective modeled mineralized volume, dimensionless)
Contained graphite is:

$$G_{r,\mathrm{scaled}}(c) = T_{r,\mathrm{scaled}}(c) \cdot \frac{\bar{Z}_r(c)}{100}$$

Percentile tonnages are reported only when at least 5 realizations exceed cutoff.

This scaling factor is an explicit scenario assumption for this pilot study and is not a public reporting-code resource modifier. This manuscript reports scenario-scaled tonnage in the main results and provides gross/unscaled values as supplementary derived output.

### 3.14 Scenario-basis tonnage sensitivity

To make scenario dependence explicit, scenario-scaled tonnage is reported in main text (`risked_tonnage.csv`) and unscaled gross tonnage is provided as supplementary derived output (`risked_tonnage_unscaled.csv`). Scaling is linear by formulation:

$$T_{P50}(f_v) = T_{P50,\ gross} \cdot f_v$$

where $T_{P50,\ gross}$ is the unscaled P50 mass on the modeled grid.
At 3% cutoff, the corresponding unscaled gross tonnage is 3825.79 Mt (P50) from `risked_tonnage_unscaled.csv`.

### 3.15 Random-fold cross-validation check (non-spatial)

A 5-fold ordinary kriging cross-validation was run in normal-score space with back-transform to original units. Procedure: declustered composites were randomly shuffled with seed 42, split into 5 folds, and predictions were generated fold-wise from training data. For each fold, trend coefficients and NST were refit on training samples; variogram model settings were held fixed to the run configuration.

This is a random-fold (non-spatial) check and may be optimistic under spatial autocorrelation. It is reported as an internal consistency diagnostic, not as a spatially independent validation.
A blocked spatial cross-validation is included in this manuscript package (`cross_validation_blocked_300.json`) and is used as primary predictive evidence relative to random-fold diagnostics.

The submission package includes:

- `cross_validation_300.json` (n = 300, seed 42)
- `cross_validation_600.json` (n = 600, seed 42)
- `cross_validation_blocked_300.json` (n = 300, 5 folds, blocked XY CV using 500 m blocks)

## 4. Variography Results

Directional continuity is evident in strike/down-dip directions and weakly constrained in the normal direction due to sparse lag support. For interpretation, this model should be treated as quasi-2D (strike/down-dip dominated) with provisional normal-direction continuity.

## 5. Simulation and Validation Results

### 5.1 Internal validation summary

Internal validation diagnostics are reported in Table 7. Distributional agreement is strong, while swath correlations are moderate and directionally consistent with anisotropy support. The emphasis on ensemble-scale distribution and swath behavior, rather than block-by-block matching, follows common SGS validation practice (Nowak and Verly, 2005).
Continuity checks (Table 14) show only modest swath-correlation shifts (X -0.0464, Y +0.0644, Z +0.0085), indicating no large directional distortion in these diagnostics.

### 5.2 Internal diagnostics used for reporting

Internal diagnostics used for reporting:

- Mean grade (data/sim): 4.8600% / 4.8532%
- Std (data/sim): 1.1477% / 1.1513%
- Histogram overlap: 0.9789
- QQ RMSE: 0.0760
- Swath correlations (X/Y/Z): 0.5844 / 0.6229 / 0.4123
- Swath coverage (P10-P90): 87.33%

### 5.3 Internal random-fold diagnostics

Cross-validation:
- blocked spatial CV (primary): ME = -0.1859, MAE = 2.0674, RMSE = 2.6153
- 300-sample run: ME = -0.0661, MAE = 1.7358, RMSE = 2.2823
- 600-sample run: ME = 0.0408, MAE = 1.5430, RMSE = 2.1272

Relative to data standard deviation (1.1477), RMSE is 2.28x (blocked), 1.99x (300-sample random-fold), and 1.85x (600-sample random-fold). This is not a small error regime; point-support CV is intentionally harsher than distributional histogram/QQ checks and is interpreted as evidence of substantial local predictive uncertainty. Accordingly, CV is framed here as predictive-difficulty evidence at point support, while the primary project objective remains pilot-scale uncertainty screening and risk bracketing rather than high-precision point prediction.

### 5.4 Variogram reproduction check (realizations vs target)

To address SGS-standard structure validation, a lag-wise variogram reproduction check was run for along-strike and down-dip directions using the final realization ensemble (`outputs/tables/variogram_reproduction_lag.csv`). Summary statistics (Table 16) indicate moderate reproduction error in grade space at pilot support, with the shortest lag affected by coarse 100 m block spacing and limited short-distance pair support. This result is treated as consistent with screening-scale behavior rather than a claim of full variogram reproduction at all lags.

## 6. Risk Analysis

Table 6 presents scenario-scaled pilot-grid methodological output only; it is not a Mineral Resource or Ore Reserve statement and is not code-compliant resource reporting. To reduce resource-style presentation risk, the main text reports only the primary screening cutoff; full cutoff sweeps are provided in supplementary outputs (`risked_tonnage.csv` and derived unscaled `risked_tonnage_unscaled.csv`).

At 3% TGC cutoff, risked tonnage is:

- P10: 209.52 Mt
- P50: 210.53 Mt
- P90: 211.18 Mt
- P50 grade: 4.91% TGC
- P50 contained graphite: 10342 kt
- Unscaled gross P50 tonnage (same cutoff): 3825.79 Mt (`risked_tonnage_unscaled.csv`)

These values are pilot-grid uncertainty outputs only, not reportable resource figures.
Normal-range sensitivity (Table 15; best-fit configured normal range = 70 m, with sensitivity references at 60 m and 150 m) shows modest impact on Z-swath correlation and minimal impact on mean probability and 3% cutoff P50 tonnage.

## 7. Discussion and Future Work

### 7.1 Robust findings versus provisional elements

Directional anisotropy and pilot-scale risk outputs are robust in strike and down-dip directions under the current data support. Normal-direction continuity remains the most provisional component because pair support is sparse at larger lags and should be interpreted with explicit caution in planning contexts.

### 7.2 Comparison to related mining precedent

The present workflow is aligned with established SGS practice and is consistent with African case-study precedent where probabilistic grade-tonnage framing is used for uncertainty communication (Nowak and Verly, 2005; Paithankar and Chatterjee, 2018). In this sense, the contribution here is a reproducible pilot-screen implementation for stratiform graphite, not a claim of reporting-code resource declaration.

### 7.3 Two-stage uncertainty and computational scale-up

A practical next step is two-stage uncertainty treatment that separates geological geometry/domain uncertainty from grade uncertainty before integrated decision metrics, as demonstrated in related DS/MPS+SGS uncertainty studies (van der Grijp and Minnitt, 2015; Paithankar and Chatterjee, 2018). For larger scenario campaigns, machine-learning-assisted SGS acceleration is also relevant to reduce turnaround while preserving uncertainty diagnostics (Bai and Tahmasebi, 2022).

## 8. Limitations

1. Normal-direction variogram support is weak at higher lags.
2. Calibration improves distributional fit by design and should not be interpreted as independent predictive proof.
3. Random-fold cross-validation is optimistic relative to blocked spatial CV.
4. This is a pilot grid with coarse support and is not a mine-planning block model.
5. Independent `f_v` derivation is pending and required before publication.

## 9. Conclusions

1. The workflow is reproducible and documents desurvey, compositing, declustering, variography, SGS, trend handling, internal validation, and risk postprocessing.
2. Directional anisotropy is evident in strike/down-dip directions, with lower confidence in the normal direction.
3. The 200-realization ensemble provides pilot-scale uncertainty products for screening.
4. Under the explicit pilot-screen scenario factor ($f_v = 0.055$), at 3% TGC cutoff risked tonnage is 209.52/210.53/211.18 Mt (P10/P50/P90) with P50 grade 4.91%.
5. Outputs support uncertainty screening and drill-prioritization only; they are not code-compliant resource declarations.

## Data Availability

All scripts and outputs used for this manuscript are included in this submission package under `repo/`, `src/`, and `outputs/` (including `validation_metrics_pre.json`, `cross_validation_300.json`, and `cross_validation_600.json`). Reproducible execution entry point: `python -m src.run_all --config config/project_best_fit.yaml --output outputs`.

Reproducibility manifest for reported tables/figures:

- Table 6: `risked_tonnage.csv`
- Table 7: `validation_metrics_pre.json`, `validation_metrics.json`
- Table 8: `cross_validation_300.json`, `cross_validation_600.json`
- Table 8 blocked CV: `cross_validation_blocked_300.json`
- Table 9: `variogram_pair_counts.csv`
- Table 10: `trend_diagnostics.json`
- Table 12: `stationarity_trends.json`, `trend_stationarity_reconciliation.json`
- Table 14: `validation_metrics_pre.json`, `validation_metrics.json`
- Table 15: `normal_range_sensitivity.csv` (from `outputs/`, `outputs_normal_60/`, `outputs_normal_150/`)
- Table 16: `variogram_reproduction_summary.json`, `variogram_reproduction_lag.csv`
- Supplementary unscaled tonnage: `risked_tonnage_unscaled.csv`
- Figure 1: `outputs/figures/variogram.png`, `outputs/figures/variogram_model.json`
- Figures 2-4: `outputs/figures/histogram_validation.png`, `outputs/figures/qq_plot.png`, `outputs/figures/swath_x.png`, `outputs/figures/swath_y.png`, `outputs/figures/swath_z.png`
- Figure 5: `outputs/figures/trend_diagnostic.png`
- Figure 6: `outputs/figures/composite_length_hist.png`

A public DOI-backed archive and immutable URL are not included in this package and must be assigned before publication.

## References

Deutsch, C.V., Journel, A.G., 1998. GSLIB: Geostatistical Software Library and User's Guide, 2nd ed. Oxford University Press.

Goovaerts, P., 1997. Geostatistics for Natural Resources Evaluation. Oxford University Press.

Chiles, J.-P., Delfiner, P., 2012. Geostatistics: Modeling Spatial Uncertainty, 2nd ed. Wiley.

Journel, A.G., Huijbregts, C.J., 1978. Mining Geostatistics. Academic Press.

Remy, N., Boucher, A., Wu, J., 2009. Applied Geostatistics with SGeMS. Cambridge University Press.

Isaaks, E.H., Srivastava, R.M., 1989. An Introduction to Applied Geostatistics. Oxford University Press.

Matheron, G., 1973. The intrinsic random functions and their applications. Advances in Applied Probability 5, 439-468.

Deutsch, C.V., 2002. Geostatistical Reservoir Modeling. Oxford University Press.

Nowak, M., Verly, G., 2005. The Practice of Sequential Gaussian Simulation. In: Geostatistics Banff 2004. Springer.

Zanon, S., Leuangthong, O., 2003. Selected Implementation Issues with Sequential Gaussian Simulation. CCG Annual Report.

Babak, O., 2006. Variogram Reproduction in Sequential Simulation: Interaction Between Screening and Search Strategy. CCG Technical Report.

Paithankar, A., Chatterjee, S., 2018. Grade and Tonnage Uncertainty Analysis of an African Copper Deposit Using Multiple-Point Geostatistics and Sequential Gaussian Simulation. Natural Resources Research 27(4), 419-436.

van der Grijp, Y., Minnitt, R.C.A., 2015. Application of Direct Sampling multi-point statistic and sequential Gaussian simulation algorithms for modelling uncertainty in gold deposits. Journal of the Southern African Institute of Mining and Metallurgy 115(1).

Bai, T., Tahmasebi, P., 2022. Sequential Gaussian simulation for geosystems modeling: A machine learning approach. Geoscience Frontiers 13(1), 101258.


