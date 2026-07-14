# Tables (Source-of-Truth Generated)

## Table 1. Data and Processing Audit

| Processing stage | Holes | Records | Meters | Purpose |
|---|---:|---:|---:|---|
| Drillhole policy | 100 used; 100 validated | - | - | Study scope. |
| Raw assays | 100 | 3,350 | 7902.37 | TGC source. |
| Analytical QA/QC controls | - | 373 | - | 93 CRMs, 94 blanks, 93 coarse duplicates and 93 pulp duplicates; accepted batch review. |
| Lithology logs | 100 | 1,248 | 9416.90 | Geology source. |
| Desurveyed assays | 100 | 3,350 | 7902.37 | XYZ support. |
| 2 m composites | 100 | 4,129 | 7957.70 | Nominal span; 7878.28 m assay-covered and 79.42 m internal-gap sensitivity across 88 composites. |
| Domain composites | 100 | 4,129 | 7957.70 | SGS input: 3,566/382/181. |
| Geological domains | 100 | 3,566 fresh graphitic; 382 weathered graphitic; 181 host/waste | - | Graphitic-only weathering contrast and host/waste control. |
| Online Resource 2 | - | 13 worksheets | - | Audit-level run metadata, validation, variogram, convergence, support, contact, occupancy and repeated-null summaries. |

## Table 2. Domain and Grade Summary

| Group | n | Mean TGC (%) | Median TGC (%) | Standard deviation (%) | Composites at or above 3% (%) | Basis |
|---|---:|---:|---:|---:|---:|---|
| fresh graphitic domain | 3566 | 4.212 | 3.910 | 2.231 | 65.90 | 2 m composites |
| host/waste domain | 181 | 1.026 | 0.510 | 1.492 | 6.08 | 2 m composites |
| weathered graphitic domain | 382 | 4.801 | 4.635 | 2.573 | 73.04 | 2 m composites |

## Table 3. Simulation and Variogram Configuration

| Item | Value |
|---|---|
| Simulation and reporting support | 25 x 25 x 2 m simulation; 50 x 50 x 2 m reporting |
| Ensemble | 100 realisations; seed 1337 |
| Local estimator | simple-kriging-style conditional estimator in domain-wise normal-score space; Online Resource 2 reports the implemented estimator and retains the legacy configured label only as provenance |
| Categorical domains | fresh graphitic, weathered graphitic and host/waste; local inverse-distance class scores in a 250/200/20 m ellipsoid, maximum 20 neighbours, prior weight 2.0; seed rule 1337 + realisation index |
| Geological threshold and top cut | 3% TGC is a screening threshold (composite Q25 2.358%, median 3.849%); no top cut applied, with 99.5th-percentile sensitivity affecting 19 composites |
| Boundary treatment | categorical domains vary between realisations but are hard for grade conditioning within each paired realisation |
| Structural axes and search | strike 000/180 degrees; down dip 090 degrees at 30 degrees; normal 270 degrees; radii 250/200/20 m |
| Grade neighbourhood | fixed minimum/maximum 3/20; simulated nodes enter the conditioning search |
| Variogram | exponential; range parameter 250 m; nugget 0.20; structured sill 0.80; 50 m lags, 10 lags, 500 m maximum distance and 22.5 degrees tolerance |
| Declustering | 200 x 200 x 5 m cells; 100/200/300 m XY cells at 5 m Z give all-composite means 3.936/3.794/3.800% TGC; graphitic-only means 4.070/3.921/3.926% TGC. |
| Numerical mean check | full grid 2.056%; fractional envelope 3.829%; full-cell core 3.903%; declustered graphitic composites 3.921% TGC |
| Validation scope | histogram/Q-Q, support-matched swaths, variogram envelopes, ensemble stability, hole-grouped categorical reliability/confusion, 20-versus-20 null sensitivity, null-realisation bootstrap and withheld-composite baselines; no independent blocked rerun of the final SGS ensemble |

## Table 4. Validation and Information-Content Comparison

| Validation axis | Geology-conditioned evidence | Matched reference or null evidence | Supported interpretation |
|---|---|---|---|
| Archive-derived reporting support | 55,716 common 25 x 25 x 2 m blocks; fractional volume 4.313% | six retained lode IDs; L01 contributes 92.99% | Common support is explicit; evidence primarily represents L01 |
| Support-aligned means | full grid 2.056%; fractional envelope 3.829%; core 3.903% TGC | declustered graphitic composites 3.921% TGC | Full-grid and graphitic-support means answer different volume questions |
| Matched 20-versus-20 envelope comparison | mean 3.825 (3.814-3.863)%; P(TGC > 3%) 0.656 (0.654-0.665); spread 3.199 (3.183-3.233)%; histogram overlap 0.876 (0.873-0.877); Q-Q RMSE 0.474 (0.459-0.475)% | null mean 3.892 (3.880-3.925)%; P(TGC > 3%) 0.595 (0.593-0.600); spread 5.068 (5.004-5.088)%; overlap 0.925 (0.924-0.928); Q-Q RMSE 0.233 (0.218-0.239)% | Null retains closer marginal fit; conditioning gives higher persistence and narrower conditional spread on identical support |
| Envelope-aligned directional swaths | canonical subset median strike/down-dip/normal r 0.761/0.822/0.834 | null median 0.734/0.857/0.856 | Directional reproduction is mixed; no overall model winner is assigned |
| Spatial support pattern | high-spread median nearest-composite distance 90.3 m; 32.5% on footprint edge | persistent-occupancy median distance 40.4 m; joint set 20 columns | Separates cross-package/contact follow-up from recurrent graphitic support |
| Ensemble and variogram behaviour | n=75 probability MAE 0.018; probability r 0.995; spread r 0.942; full-ensemble swath r 0.777/0.823/0.840 | matched-space variogram weighted RMSE 0.237; thickness-normal direction has two pair-supported lags | Quantifies Monte Carlo and covariance behaviour on selected support |
| Categorical and withheld validation | macro-F1 0.356; balanced accuracy 0.442; graphitic-host ROC-AUC 0.708 | within-support Brier skill -0.407; 500 m block/leave-hole/leave-section grade RMSE 2.261/2.179/2.232% TGC | Categorical fields rank relative patterns; withheld baselines bound local predictive evidence |

## Table 5. Practical Decision-Use Matrix for Graphite Exploration and Resource Evaluation

| Product | Geological meaning | Evidence used | Practical use |
|---|---|---|---|
| Archive-derived lode envelope | Common volume for the interpreted graphitic corridor | Exact support alignment, common-footprint and DEM checks | Keep grade summaries tied to an explicit geological volume |
| Persistent P(TGC > 3%) | Above-threshold support recurring across realisations | Completed ensemble, n=75 stability and distance-to-support analysis | Identify corridor segments suitable for step-out confirmation |
| P90-P10 TGC spread | Conditional grade range inside the envelope | Ensemble convergence, variogram reproduction and directional swaths | Target infill sampling where grade remains variable |
| Joint persistence and high spread | Graphitic support persists while conditional grade remains broad | Upper-decile spread and P >= 0.80 co-location | Prioritise section review and holes oriented across the package |
| Raw categorical frequencies and entropy | Relative ambiguity in the archived local class scorer | Hole-grouped validation within mapped search support | Guide re-logging and contact review; keep absolute class calibration separate |
| Matched geology-blind comparison | Behaviour of an alternate configuration inside the same lode volume | Five independent null seeds and five canonical 20-realisation subsets | Evaluate distribution fit and geological organisation on separate axes |
