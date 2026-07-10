# Tables (Source-of-Truth Generated)

## Table 1. Data and Processing Audit

| Processing stage | Holes | Records | Meters | Purpose |
|---|---:|---:|---:|---|
| Drillhole policy | 100 used; 100 validated | - | - | Study scope. |
| Raw assays | 100 | 3,350 | 7902.37 | TGC source. |
| Lithology logs | 100 | 1,248 | 9416.90 | Geology source. |
| Desurveyed assays | 100 | 3,350 | 7902.37 | XYZ support. |
| 2 m composites | 100 | 4,129 | 7957.70 | Composite support. |
| Domain composites | 100 | 4,129 | 7957.70 | SGS input: 3,566/382/181. |
| Geological domains | 100 | 3,566 fresh graphitic; 382 weathered graphitic; 181 host/waste | - | Graphitic-only weathering contrast and host/waste control. |
| Online Resource 2 | - | 11 worksheets | - | Audit-level run metadata, validation, variogram, convergence, support, contact, occupancy and repeated-null summaries. |

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
| Numerical mean check | data 4.127% TGC; whole reporting-support SGS 2.056%; graphitic-probability >=0.70 cells 3.704%; host-probability >=0.70 cells 1.134% |
| Validation scope | histogram/Q-Q, support-matched swaths, variogram envelopes, ensemble stability, hole-grouped categorical reliability/confusion, 20-versus-20 null sensitivity, null-realisation bootstrap and withheld-composite baselines; no independent blocked rerun of the final SGS ensemble |

## Table 4. Validation and Information-Content Comparison

| Validation axis | Geology-conditioned evidence | Null or reference comparison | Supported interpretation |
|---|---|---|---|
| Archive-derived reporting support | 55,716 common 25 x 25 x 2 m blocks; fractional lode volume 4.313% | any-intersection 19,286 reporting cells; full-cell core 8,938 cells | Support sensitivity only; the seven-lode mask shares project data and is not independent validation |
| Support-aligned ensemble means | full box 2.056%; any lode cell 3.783%; fractional lode volume 3.829%; core 3.903% TGC | declustered graphitic composites 3.921% TGC | The apparent full-grid deficit is a reporting-support difference, not direct local validation |
| Envelope ensemble stability | n=75 probability MAE 0.018; probability r 0.995; spread r 0.942 | 100-realisation envelope reference | Numerical stability at selected reporting support, not predictive calibration |
| Variogram and envelope-aligned swaths | matched-space variogram weighted RMSE 0.237; graphitic-composite versus envelope P50 swath r 0.777/0.823/0.840 | pair-limited thickness-normal direction retained as caveat | Tests covariance and directional behaviour on matched reporting support |
| Categorical sensitivity | macro-F1 0.356; balanced accuracy 0.442; graphitic-host ROC-AUC 0.708; raw Brier skill -4.896 | grouped whole-hole folds with zero leakage | Raw categorical products are secondary sensitivity diagnostics, not calibrated reporting boundaries |
| Full-grid null sensitivity | canonical histogram overlap 0.602; Q-Q RMSE 2.144 | five independent no-domain seed families in Online Resource 2 | Global fit and support-aligned interpretation remain separate evaluation axes |
| Withheld grade baselines | 500 m block/leave-hole/leave-section RMSE 2.261/2.179/2.232% TGC | simple spatial estimators under held-out support | Bounds local prediction; no blocked rerun of final SGS is claimed |

## Table 5. Practical Decision-Use Matrix for Graphite Exploration and Resource Evaluation

| Product | Geological meaning | Validation support | Appropriate use |
|---|---|---|---|
| Archive-derived lode envelope | Common reporting support for the interpreted graphitic corridor | Exact grid alignment, common-footprint and DEM-surface checks | Compare completed SGS summaries inside versus outside the project-derived envelope |
| Envelope-weighted P50 TGC | Central conditional grade behaviour inside the selected support | Completed ensemble and support-sensitivity brackets | Compare broad graphitic-support sections; not a local grade prediction |
| P90-P10 TGC spread | Conditional grade range within the selected support | Envelope convergence, variogram reproduction and directional swaths | Identify relative grade-uncertainty zones for geological follow-up |
| P(TGC > 3%) | Modelled above-threshold occupancy inside the envelope | Completed ensemble and support-aligned maps | Compare relative persistence of modelled above-threshold support |
| Raw categorical frequencies and entropy | Sensitivity of the local categorical scoring rule | Grouped categorical validation in Online Resource 2 | Audit relative model ambiguity only; do not treat as calibrated boundary probability |
| Geology-blind null sensitivity | Global-distribution behaviour under a composite alternate configuration | Five independent 20-realisation families | Prevent model choice from relying only on histogram or Q-Q fit |
