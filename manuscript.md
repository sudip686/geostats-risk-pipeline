# Geological Support and Reporting-Envelope Effects on Grade Uncertainty in a Tanzanian Stratiform Graphite System

**Authors:** Sudipta Chanda

**Affiliations:** Sakariya Mines and Minerals Private Limited, 1402 Ecostation Business Tower, Newtown, Rajarhat, Kolkata, West Bengal 700160, India

**Corresponding author:** Sudipta Chanda

**Corresponding author email:** sudipta.chanda@sakariya.in

**ORCID iD:** 0009-0001-5030-7524

**ORCID record:** https://orcid.org/0009-0001-5030-7524

**Acknowledgements:** The author acknowledges the technical contributions of project team members who supported data preparation, workflow execution, and manuscript quality control.

**Author contributions:** Sudipta Chanda: conceptualization, data curation, methodology, software, formal analysis, visualization, writing - original draft, and writing - review and editing.

------------------------------------------------------------------------

## Abstract
Layer-parallel graphitic schist can define a convincing exploration corridor while leaving the volume used to summarize grade uncertainty poorly specified. We evaluate a 100-realisation, geology-conditioned Sequential Gaussian Simulation ensemble for a Tanzanian stratiform graphite system and then restrict completed outputs to a topography-clipped, archive-derived lode mask at common 50 x 50 x 2 m reporting support. The mask retains 55,716 25 x 25 x 2 m blocks (4.313% of the reporting volume). The full rectangular-grid ensemble mean is 2.056% total graphitic carbon (TGC), whereas the fractional lode-volume mean is 3.829% and the full-cell lode-core mean is 3.903%, compared with 3.921% for declustered graphitic composites. This is a reporting-support result, not independent grade validation, because the archive mask shares project lithology and threshold information with the SGS inputs. Within the envelope, probability and spread products are evaluated by convergence, variogram-reproduction and directional-swath diagnostics. Five repeated geology-blind sensitivity families retain closer selected global distribution fits, showing that global fit and geological reporting support answer different questions. The study provides a practical framework for separating support choice, conditional grade spread and model-behaviour uncertainty in graphite exploration.

**Keywords:** graphite; conditional simulation; reporting support; geological uncertainty; exploration evaluation; Tanzania

## 1. Introduction

Graphite-bearing metasedimentary horizons in the Tanzanian Mozambique Belt commonly follow compositional layering and metamorphic fabric, but this continuity does not establish the certainty of grade, contact position, weathering state or package geometry between drillholes. That distinction matters in graphite exploration because an apparently coherent graphitic corridor can be summarized over very different geological volumes.

Published work has established the regional setting, host rocks and mineralogical context of Tanzanian graphite occurrences [1, 2, 3]. A remaining mining-geology problem is how a completed simulation ensemble should be reported when its rectangular computational grid contains both graphitic-support and background volume.

Conditional simulation is useful here because it transfers uncertainty across multiple conditional outcomes rather than supplying one preferred grade surface [4]. Metallic-deposit studies have also shown that domain representation and support can change the apparent behaviour of grade uncertainty [5, 6]. Stratiform graphite has received less attention on this specific reporting-support problem.

This study asks: (1) how strongly does reporting support alter ensemble grade summaries; (2) what conditional grade spread remains inside an archive-derived graphitic lode envelope; (3) how do global-fit diagnostics from a geology-blind sensitivity compare with support-aligned diagnostics; and (4) which products can guide relative geological follow-up without claiming local predictive calibration? The contribution is a geology-led reporting-support and uncertainty framework, not a resource estimate or a local grade-prediction model.

## 2. Geological Setting

### 2.1 Regional Mozambique Belt Framework

The study area lies in northeastern Tanzania within the Tanzanian sector of the Mozambique Belt / East African Orogen, a polyphase high-grade Neoproterozoic-Cambrian system shaped by crustal thickening, granulite-facies metamorphism, nappe emplacement and later structural reworking [7, 8, 9, 10]. Fold interference and crustal decoupling documented in northern Tanzania show that apparently continuous metamorphic packages may contain local structural complexity [11]. The mapped graphitic-schist corridor lies within this northern high-grade belt segment.

Figure 1 establishes the geological argument at three nested scales. Panel (a) locates northeastern Tanzania within the generalized East African Orogen-Mozambique Belt system, panel (b) places the study area in the eastern high-grade belt relative to the Tanzania Craton and adjacent Proterozoic belts, and panel (c) redraws the owned project mapping at drill-corridor scale with the 100 canonical collars. The mapped N-S to NNE-SSW graphitic-schist bands and adjacent khondalite/aluminous schist, mafic granulite and quartzofeldspathic units justify testing fabric-parallel continuity as a first-order prior. The map does not resolve contact position, package thickness or TGC continuity between holes; those remain the uncertainty questions evaluated by Figures 3-7.

### 2.2 Graphite mineralisation in Tanzanian Mozambique Belt terranes

Graphite mineralisation in Tanzanian Mozambique Belt terranes occurs in high-grade metasedimentary packages whose protolith composition, metamorphic fabric and later deformation organise graphite-bearing layers [1, 3, 12]. In that framework, graphite is commonly disseminated or layered in graphitic schist, mica schist, quartzofeldspathic gneiss, calc-silicate gneiss, marble or khondalite-like aluminous graphitic rocks. For the same Maramba-Tanga system, Das et al. [2] report petrographic evidence for graphite flakes aligned with foliation in graphitic schist, XRD evidence for crystalline graphite, Raman spectra with ordered graphite bands and weak defect response, and SEM/FTIR evidence for lamellar graphite with associated silicate/clay phases. Those published observations provide independent contextual support for graphitic schist, foliation-parallel fabric and weathering state as geological variables. SGS performance and uncertainty products are evaluated from this study's workflow outputs.

Graphite mineralisation in northeastern Tanzania is therefore interpreted within the wider high-grade metasedimentary framework of the Mozambique Belt. Graphite-rich pelitic and psammitic gneisses are documented in the Merelani-Lelatema area [12], and regional studies show that metasedimentary packages can be repeatedly transposed during Pan-African tectonism. The key metallogenic implication for this study is that graphitic-schist continuity, contact uncertainty, weathering state and thickness-normal continuity are the geological variables that need direct testing.

Figure 2 makes the structural hypothesis auditable by showing exactly how geology is converted into numerical continuity and search directions. Panel (a) defines the observed north-south corridor and the 000/180 degree strike proxy, panel (b) shows the east-dipping 090 degree/30 degree down-dip direction and its orthogonal plane normal, and panel (c) states the 250/200/20 m search radii. The figure is therefore a parameter-definition figure, not evidence that the imposed global anisotropy is locally correct.

The convention is selected from local geological support: graphitic intervals form a north-south drillhole/composite corridor, the logged graphitic package is treated as east dipping at study scale, published local geology reports a broadly north-south to NNE-SSW graphitic-schist trend with moderate east dip and foliation-parallel graphite [2], and the SGS needs orthogonal axes that separate fabric-parallel, down-dip and thickness-normal continuity. Figure 3 is then an observational compatibility check. Panels A and B show the spatial and elevation coverage of assayed composites along the configured axes, while panel (c) quantifies sampled metres and 3% TGC threshold occupancy along the corridor. This supports testing the first-order proxy and identifies unevenly constrained reaches, but it neither validates the SGS nor establishes continuity between drillholes.

### 2.3 Local Drillhole Geological Observations

The drillhole database records graphitic schist, khondalite, quartzite, mafic granulite, quartzo-feldspathic schist and weathered graphitic variants within a high-grade metasedimentary package. This assemblage and the graphitic-schist host relationship agree with the regional high-grade metasedimentary framework and the original geological synthesis in Figure 1. Published weathered regolithic/kaolinised material and associated major-oxide patterns also support treating weathering as a geological state rather than only a modelling label.

The local assay data then quantify how those logged categories relate to TGC. Table 1 defines the drillhole, assay, lithology and composite populations, while Table 2 shows that graphitic-coded composites have a median of 3.94% TGC and 66.37% of records at or above 3% TGC, compared with 2.34% TGC and 43.51% for non-graphitic composites. Figure 3 adds the required spatial context: the two projections show where vertical and lateral support exists, and the corridor profile shows that sampled metres and above-threshold occupancy are unevenly distributed. That distribution explains why later SGS outputs are interpreted as uncertainty localisation rather than direct interpolation between uniformly supported drill sections. The existing representative section inventory and lode-scale summaries provide a second geology layer beyond the SGS maps: they show that graphitic support concentrates into persistent section-scale and lode-scale features, which is consistent with the corridor-scale prior used here.

Together, the mapped corridor, logged contacts and grade contrasts define the geological problem carried into SGS. They support testing stronger continuity within graphitic schist than across its boundaries, a possible local weathering-associated contrast, and shorter continuity normal to the graphitic package. They do not define fixed resource boundaries or prove that weathering causes graphite enrichment.

### 2.4 Geological Priors Tested in This Study

Four geological priors organise the analysis. The lithological prior separates fresh graphitic, weathered graphitic and host/waste categories. The contact prior tests whether uncertainty increases near graphitic-domain transitions. The weathering prior tests a distributional contrast within graphitic composites without assuming a causal enrichment mechanism. The structural prior tests whether continuity is more coherent along a first-order fabric-concordant ellipsoid than normal to the modelled package.

Each prior maps to a numerical control described in Methods. Two-metre compositing and vertical cells preserve logged contact scale; hard domains preserve lithology and weathering categories; the 3% TGC threshold identifies above-threshold model occupancy; the 000/090/270 degree convention provides the structural axes; the 250/200/20 m frame limits thickness-normal conditioning; and the 3/20 neighbourhood controls local data influence. These controls were fixed before interpretation and are evaluated through model behaviour rather than selected post hoc for the closest histogram.

## 3. Data and Methods

### 3.1 Drillhole Database and Workflow Quality Assurance and Quality Control (QA/QC)

The study uses 100 drillholes, 3,350 assay intervals and 1,248 lithology records. Table 1 follows the data from raw assays and lithology logs through desurveying, compositing, domain assignment and the audit-level Online Resource 2 workbook. The quantitative analyses are based on the curated project database and reproducible workflow outputs. Before modelling, the workflow checks interval validity, assay/lithology support, survey availability and the 100-hole study policy; four holes with incomplete assay/lithology support are excluded from study metrics. The QA/QC statement in this manuscript therefore refers to the reproducible workflow's data-integrity checks and audit trail, not to an independent validation of the SGS ensemble.

### 3.2 Compositing and Support

Assay intervals were desurveyed and composited to 2 m support using length weighting:

Equation (1):

```math
Z_{\mathrm{comp}} = \frac{\sum_i L_i Z_i}{\sum_i L_i}
```

where composite TGC is calculated from the length and TGC of each contributing assay interval. The 2 m length regularises variable assay intervals while retaining the vertical scale of logged graphitic and weathering contacts. A minimum retained length of 0.5 m prevents short edge intervals from being forced into artificial 2 m composites.

Simulation used 25 m x 25 m x 2 m cells. The lateral dimensions retain plan-view graphitic-body morphology without treating individual assays as mappable panels between drill sections, while the 2 m vertical dimension matches composite support and preserves contact/weathering resolution. This support represents the geological scale tested; local predictive behaviour is evaluated separately by the validation diagnostics.

Results were aggregated to 50 m x 50 m x 2 m reporting support. The two-by-two lateral aggregation is tied to the closest local drill spacing and stabilises map-scale probability and spread diagnostics while retaining the 2 m vertical dimension. It is an averaging support for comparison and visualisation, not evidence that every 50 m block is independently predicted.

No top-cut was applied because the graphitic 2 m population does not contain a detached high-grade tail. It has n = 3,948, mean 4.27% TGC, median 3.96% TGC, maximum 14.67% TGC and COV 0.53. A 99.5th-percentile cap at 11.99% TGC would affect 19 composites (0.48%) and change the mean and variance by only -0.13% and -1.88%, respectively.

### 3.3 Domain Definition

Composites were grouped into fresh graphitic, weathered graphitic and host/waste categories. The 3% TGC threshold was selected from the study dataset before SGS as a geological screening threshold separating weakly graphitic/background material from more continuous graphitic-schist support in the local histogram and logs. In the canonical 2 m composites it lies between the lower quartile (2.358% TGC) and median (3.849% TGC), and 64.26% of composite metres are at or above it. The threshold is used only for domain checks and above-threshold model occupancy; no economic assumptions are applied. Table 3 records the threshold, boundary, search-neighbourhood and variogram settings. The hard-boundary case is a geological-prior end member that prevents grade conditioning across fresh graphitic, weathered graphitic and host/waste categories within each realisation.

### 3.4 Categorical Sensitivity Used by Grade SGS

Fresh graphitic, weathered graphitic and host/waste classes were assigned from logged composites and sampled from fixed local inverse-distance probability scores within the configured anisotropic search. The archived implementation draws categories independently at grid nodes; it is not indicator SGS, a transition-probability simulation or a spatially coherent geological-body model. Grade SGS was then performed within the sampled class structure. Raw class frequency and entropy products are retained in Online Resource 2 as secondary model-sensitivity diagnostics. They are not used to define the primary reporting envelope or interpreted as calibrated class probabilities.

### 3.5 Declustering and Normal-Score Transformation (NST)

Cell declustering used 200 m x 200 m x 5 m cells. The X-Y cell size was selected to reduce drillhole-cluster bias at approximately the scale of the broader drill spacing while preserving the 2 m composite support in the vertical direction through a 5 m declustering height. A sensitivity test of cell size supports the choice: 100/200/300 m XY cells at 5 m Z give all-composite means 3.936/3.794/3.800% TGC; graphitic-only means 4.070/3.921/3.926% TGC. The 200 m result is therefore used as a conservative declustered reference rather than as a tuned parameter. NST was applied before SGS and back-transformed to TGC units after simulation, following standard geostatistical support and distribution handling [13, 14, 15, 16]. 

### 3.6 Variography and Structural Prior

Directional variography tested continuity along strike, down dip and normal to the graphitic package using 50 m lags, 10 lags, a 500 m maximum distance and 22.5 degrees directional tolerance. The 50 m lag is close to the reporting-map support and local infill spacing, while the 500 m window tests continuity over approximately two major-range lengths. Pair-count support is strongest in the along-strike and down-dip directions and sparse normal to plane: pair totals along/down/normal = 205,965/57,295/26,911; normal-to-plane has 4/10 nonzero lags. The final SGS model used one exponential structure, one nugget interpretation in normal-score space, a 250 m major range parameter, nugget 0.20 and structured sill 0.80. The configured directional ranges and search radii are 250/200/20 m. These values implement the local geological continuity concept: longest continuity along the north-south graphitic corridor, slightly shorter down-dip continuity within the east-dipping package, and deliberately short thickness-normal continuity so the search does not smear grade across the graphitic package. The search radii are set equal to the first-order variogram-axis ranges for the fixed SGS run; they are not tuned from the validation plots.

### 3.7 Structural-Axis Convention

The geostatistical model uses an orthogonal ellipsoid defined by strike, down-dip and plane-normal directions. In this convention, the strike line is assigned an azimuth of 000 degrees, equivalent to 180 degrees as an undirected line, the down-dip vector is assigned an azimuth of 090 degrees with a dip of 30 degrees, and the plane-normal vector is represented by 270 degrees. The axis choice was made from local geological reasoning before SGS interpretation: the graphitic composite corridor is north-south elongate, the local section geometry is consistent with an east-dipping graphitic package, and a right-handed ellipsoid is needed to test layer-parallel continuity separately from thickness-normal behaviour. The axes are used as a reproducible global first-order geostatistical proxy for search and variogram calculations; local folding and lens-scale curvature are evaluated as sources of residual structural uncertainty.

The run does not use dynamic/local dip handling. That choice is deliberate for this manuscript: the auditable input package contains drillhole, assay and lithology data, but not a shareable cell-wise structural-string model from which local dip could be regenerated. A dynamic search would require azimuth, plunge and dip values assigned to each grid cell from interpreted structural strings; without those shareable inputs, it would make the published workflow less reproducible. A global axis set keeps the parameterisation reproducible and directly tests the first-order geological hypothesis. Dynamic dip, local anisotropy or structural unfolding would be a different model class and would require a full rerun and a separate validation comparison.

### 3.8 SGS

The canonical model comprises 100 conditional SGS realisations with seed 1337. The implemented local estimator is simple-kriging-style in normal-score space: covariance weights are solved without the Lagrange multiplier used by ordinary kriging. The archived run configuration contained a legacy `OK` label, but the Online Resource 2 metadata report `SK_style_effective` as the primary estimator and retain the legacy label only in a provenance field. No simulation values were changed by this metadata correction.

First-order mean contrasts are handled by hard fresh-graphitic, weathered-graphitic and host/waste domains and by domain-wise NST. No explicit deterministic grade trend or detrending correction is applied in the production SGS; elevation-related grade behaviour is retained as diagnostic geological context rather than imposed as a drift term. The production model therefore transfers residual within-domain variability around a zero-mean Gaussian framework rather than imposing a deterministic grade trend.

The search ellipsoid and neighbourhood were fixed before SGS interpretation. Minimum/maximum neighbours of 3/20 provide a small local conditioning set while limiting dense-cluster influence, and simulated nodes are added immediately to the conditioning search. The implementation does not impose a minimum number of distinct drillholes per node, so the neighbourhood controls local sample count rather than guaranteeing balanced drillhole support.

One hundred realisations provide the ensemble for percentile, exceedance-probability, entropy and cutoff-occupancy summaries. SGS is used here as an uncertainty-transfer mechanism: it carries the lithological domains, first-order structural proxy and fitted covariance model into multiple conditional outcomes, then reveals where those controls fail to constrain a narrow range of plausible values. It is not treated as an exact local grade predictor.

The completed trend-disabled ensemble has a reporting-support minimum of 0.000% TGC and 0.00% negative cells. A zero-floor audit is therefore numerically inactive for the canonical run. This confirms physically admissible output values for the canonical run.

### 3.9 Archive-Derived Lode Envelope and Model-Behaviour Diagnostics

An archive-derived seven-lode mask was examined only as a reporting-support sensitivity. The available mask was generated algorithmically from the overlapping project database using graphitic-coded 2 m composites, a 3% TGC screening rule, fixed gap rules, spatial clustering, roof-and-floor interpolation and block-derived mesh construction. It is not the controlling MRE's unavailable 28 section-interpreted wireframes, and it does not provide independent geological validation. Its simplified construction also uses collar X/Y and collar elevation minus downhole depth rather than the desurveyed interval positions used by the SGS workflow.

Only x, y, z, block dimensions, lode identity and the DEM-derived topography field were read from the archived block table. Estimated TGC, kriging variance, neighbourhood counts, density, classification, tonnes and contained graphite were excluded. The 25 x 25 x 2 m blocks were mapped by exact centre alignment to the canonical EPSG:32737 grid. Of 57,172 archived blocks, 55,724 lie inside the SGS grid; 8 centres above the archived surface were removed, leaving 55,716 common-support blocks. The 1,448 blocks outside the SGS footprint are not compared.

At reporting support, each 50 x 50 x 2 m cell receives a fractional lode weight f equal to its retained fine-block count divided by four. Primary scalar summaries use the volume-weighted realisation mean, \(ar{Z}_r = \sum_i f_i Z_{ri} / \sum_i f_i\). Any-intersection (f > 0) and full-cell-core (f = 1) summaries provide predeclared sensitivity brackets. The original rectangular-grid summaries remain audit comparators. A vertical sum of mask occupancy is labelled vertical envelope occupancy, not true geological thickness. Histogram, Q-Q, variogram and swath diagnostics are reported as model behaviour; no independent blocked validation of the final SGS ensemble is claimed.

### 3.10 Geology-Blind Composite Null Sensitivity

Five independent no-domain isotropic families were completed with seeds 9101, 9201, 9301, 9401 and 9501, each containing 20 realisations. The null runs use direct 50 x 50 x 2 m simulation, no hard or categorical domains, one whole-population normal-score transform, isotropic 150 x 150 x 150 m covariance and search, legacy 105/15/195 degree axis labels, 8-24 neighbours, and an enabled vertical trend. The canonical ensemble uses 25 x 25 x 2 m simulation aggregated to the same 50 x 50 x 2 m reporting support, stochastic hard domains, domain-wise transforms, 250 x 200 x 20 m geological-axis covariance and search, 3-20 neighbours, and no grade trend. The comparison is therefore matched at reporting support and realisation count but is a composite configuration sensitivity rather than a one-factor domain ablation.

Five non-overlapping 20-realisation subsets of the canonical ensemble provide the realisation-count comparison. The original null family was also resampled 200 times with replacement (seed 20260707) to separate within-family Monte Carlo variation from between-seed variation. Histogram overlap, Q-Q RMSE, mean and standard deviation of simulated TGC, directional swath correlations, and swath coverage are reported for every seed; no run was selected by performance.

### 3.11 Withheld-Composite Validation Baseline



The categorical probabilities were validated independently of grade SGS by five-fold grouped cross-validation (CV). Complete drillholes were withheld, the fixed local-probability algorithm was recomputed from the remaining holes only, and predictions were evaluated at every withheld composite. Multiclass performance is reported using macro-F1, balanced accuracy, log loss and a three-class confusion matrix. Fresh plus weathered graphitic classes were combined against host/waste to calculate ROC-AUC, Brier score, Brier skill relative to each fold's training prevalence and ten fixed-width reliability bins. Predictions were separated into locations with at least one retained-hole composite inside the configured anisotropic search and locations invoking the deterministic host/waste no-support fallback. Normalised Shannon entropy was tested as a relative error-ranking score using ROC-AUC against the held-out class-error indicator. A leakage-free calibration sensitivity used four-fold grouped inner predictions to fit a logistic mapping within each outer training fold before application to its withheld holes; this mapping was not applied to the archived categorical realisations. Fold construction uses seed 20260707 and requires zero drillhole overlap.

Directional swaths in Figure 7 were computed in the configured strike, down-dip and thickness-normal coordinate system. For each of ten equal-width bins, observed composite TGC is shown only when at least five composites are present. Each realisation was averaged over reporting cells in the same bin, and the plotted P10, P50 and P90 curves are percentiles across those 100 realisation-level bin means. Aligned bars beneath each swath report the observed composite count, and a dashed line marks the five-composite display threshold. Separating sample support from the grade curves allows directional agreement, envelope width and data density to be read together without annotation obscuring the observations.

A separate withheld-composite validation baseline was run to test the predictive behaviour of the geological prior without relabelling the final SGS diagnostics as independent validation. The baseline used a reproducible 1,800-composite subset and three fold families: 500 m XY spatial blocks, leave-hole-out folds and 100 m leave-section-out folds. In each fold, inverse-distance weighting, ordinary kriging and simple kriging were trained on the retained composites and evaluated against withheld composite TGC in original units. This is a baseline validation of spatial prediction under the same geological data support, not a blocked rerun of the final 100-realisation SGS ensemble.

Run reproducibility is recorded in Online Resource 2 through the simulation seed, categorical seed rule, CRS, grid origin and support, structural axes, ellipsoidal search radii, neighbourhood limits, variogram parameters, estimator implementation and validation seeds. These fields reproduce the numerical configuration and calculation logic, while full regeneration remains conditional on access to the proprietary drillhole and categorical arrays.

### 3.12 Above-Threshold Occupancy Diagnostics

For each realisation and TGC threshold, above-threshold model occupancy was calculated from the reporting-support cells meeting that threshold and summarised across the ensemble. The complete threshold sweep is provided in `cutoff_occupancy_uncertainty.csv` in Online Resource 2. It is a screening-stage geological diagnostic used only to compare the stability of model occupancy across realisations; no density, tonnage or economic interpretation is applied in the manuscript.

### 3.13 Generative AI-Assisted Preparation and Verification

OpenAI Codex was used to assist with editorial restructuring, reference-format conversion, workflow documentation, and review of deterministic plotting and packaging code. It was not used to generate scientific images or replace geological interpretation. All numerical results were calculated from project data by the documented code, and every manuscript statement, table, figure, and reference was reviewed by the author, who accepts full responsibility for the submitted work.

## 4. Results

Results are reported in the same evidence order: Figure 3 shows drillhole geometry and threshold support, Figure 4 reports contact and weathering contrasts, Figures 5 and 6 show spatial uncertainty products, and Figure 7 with Table 4 reports validation diagnostics.

### 4.1 Drillhole, Lithology and Domain Summary

The processed dataset contains 3,350 assay records over 7902.37 m and 4,129 2 m composites. The length-weighted composite mean is 4.15% TGC, the declustered composite mean is 3.79% TGC, and graphitic-only composites average 4.28% TGC. Table 2 separates domain-grade evidence from later SGS outputs. Fresh graphitic composites average 4.21% TGC and weathered graphitic composites average 4.80% TGC; that contrast is tested in Section 4.5.

### 4.2 Structural and Variogram Evidence

Directional continuity is anisotropic. The experimental range proxies are 1187.5 m along strike, 90.7 m down dip and 21.9 m normal to plane; only the along-strike proxy exceeds the 500 m experimental-variogram window. Figure 2 reports the observed corridor, the three directional axes and the search radii applied by SGS, and Table 3 lists the corresponding variogram and search settings.

### 4.3 Archive-Derived Reporting-Support and Ensemble Behaviour

The archived lode mask and completed SGS grid share 25 x 25 x 2 m support and EPSG:32737 coordinates. After common-footprint and topography checks, the retained mask occupies 4.313% of reporting volume. It intersects 19,286 reporting cells and contains 8,938 full lode-core cells. The full rectangular-grid mean is 2.056% TGC; the any-intersection, fractional-volume and full-cell-core means are 3.783%, 3.829% and 3.903%, respectively. The fractional and core means differ from the declustered graphitic-composite mean by -0.092 and -0.018% TGC.

At n = 75, envelope probability MAE is 0.018, probability correlation is 0.995, and spread correlation is 0.942 relative to the 100-realisation reference. Matched-space variogram and envelope-aligned directional-swath results are shown in Figure 7. These are numerical stability and model-behaviour diagnostics on the selected reporting support.

### 4.4 Population and Physical-Domain Checks

The physical-domain audit found no negative reporting-support values in the canonical trend-disabled ensemble: the minimum is 0.000% TGC and the negative-cell proportion is 0.00%. Replacing negative values with zero therefore leaves the mean, P10, P50, P90 and 3% occupancy unchanged.

### 4.5 Contact and Weathering Controls

The corrected weathering comparison is restricted to graphitic-domain composites. Weathered graphitic intervals (n = 382) average 4.80% TGC, compared with 4.21% TGC for fresh graphitic intervals (n = 3566). The mean difference is 0.59 percentage points (95% CI 0.32 to 0.86; Hedges g = 0.26; Welch p < 0.001). The hole-cluster bootstrap interval is 0.11 to 1.06 percentage points, whereas the 79-hole paired comparison is inconclusive (Wilcoxon p = 0.167).

The signed graphitic-host profile contains 711 composites around 134 contiguous transitions in 42 drillholes. Graphitic-side mean TGC exceeds host/waste-side mean TGC by 2.42 percentage points. The separate unsigned graphitic-only distance-bin comparison is nonsignificant (ANOVA p = 0.467; Kruskal-Wallis p = 0.496; Levene p = 0.747). Figure 4a displays the signed profile and hole-cluster intervals.

Figure 4b displays the fresh and weathered graphitic TGC distributions. Figure 4c replots the published XRF weathering data of Das et al. [2] as contextual evidence and is not a new project measurement or an SGS validation result.

### 4.6 Envelope-Constrained Spatial Uncertainty Products

Figure 5 maps the archive-derived lode footprint together with envelope-weighted cell P50 TGC, P90-P10 TGC spread and P(TGC > 3%). White map areas are outside the common lode-envelope support. Figure 6 carries the same mask into plan and east-west section views; the DEM-derived surface line is shown where the archived topography field is available. Fixed realisations 1, 50 and 100 display between-realisation grade variation on the same masked section. Raw categorical entropy, calibration and confusion diagnostics remain in Online Resource 2 rather than defining the main uncertainty maps.

### 4.7 Above-Threshold Occupancy Diagnostics

Above-threshold model occupancy across the threshold sweep is reported in the Occupancy Diagnostics worksheet of Online Resource 2. Figure 7 reports support-aligned means, ensemble stability, variogram reproduction and geological-axis swaths; Table 4 reports histogram, Q-Q and withheld-composite metrics.

## 5. Discussion

### 5.1 Geological Support and Reporting-Envelope Effects

The main result is that the full computational box and graphitic-support volume answer different questions. The full box includes a large background component, whereas the archive-derived mask confines summary statistics to a project-interpreted graphitic corridor. The difference between 2.056% TGC in the full box and 3.829% TGC under fractional lode-volume weighting is therefore a support effect, not evidence that one summary is inherently more accurate.

### 5.2 What the Envelope Adds to Uncertainty Interpretation

Within the common envelope, probability and TGC-spread maps identify where the completed ensemble expresses persistent above-threshold occupancy and where its grade range remains broad. These maps are useful for comparing relative geological follow-up priorities inside the interpreted corridor. The vertical occupancy display is deliberately not treated as a true-thickness estimate, because the available mask is a block representation and may include disconnected vertical intervals.

### 5.3 Global Fit of the Geology-Blind Model

The five independent composite null families reproduce selected global metrics more closely than the conditioned ensemble. Median histogram overlap is 0.870 (range 0.867-0.876) versus 0.602; median Q-Q RMSE is 0.672 (range 0.558-0.718) versus 2.144. Median X/Y/Z swath correlations are 0.626/0.417/0.344, compared with 0.512/0.245/0.199 for the conditioned ensemble. Median swath coverage is 99.705% versus 40.439%, reflecting the broader spatial coverage produced by the composite null configuration. Repetition across all five seeds establishes the robustness of this global-fit behaviour and supports evaluating global distribution fit separately from geological information content.

### 5.4 Categorical Sensitivity and Geological Information

The categorical workflow remains informative as a modelling sensitivity, but whole-hole validation does not support calibrated class probabilities or independent fresh-weathered separation. The archive-derived envelope therefore carries the primary reporting-support role, while raw categorical frequency and entropy remain secondary evidence about how the local scoring rule partitions the completed simulation. The approach preserves an explicit distinction between model-implied spatial patterns and externally verified geology.

### 5.5 Implications for Graphite Exploration and Resource Evaluation

The reporting envelope provides a practical sequence for geological follow-up. The lode-support map shows where simulation summaries are being compared with an interpreted graphitic corridor. Envelope-weighted P50 TGC communicates central grade behaviour on that support, P90-P10 TGC spread identifies relative grade uncertainty, and P(TGC > 3%) shows modelled above-threshold occupancy. The full-grid comparison prevents those products from being judged only by a global histogram. These are relative geological follow-up products; they do not optimise drilling, classify resources or establish product quality.

### 5.6 Limitations and Future Validation

The archive-derived mask shares drillholes, graphitic coding and threshold logic with the SGS inputs, uses simplified vertical run geometry, and is not the unavailable 28-wireframe controlling MRE interpretation; its agreement with the SGS is therefore a reporting-support sensitivity rather than independent validation. The categorical simulator uses independent local draws, the null families change several controls together, and final-grade blocked calibration remains unavailable. Future work should test desurveyed section interpretations, spatially coherent categorical models, locally varying structure and truly independent drilling or blocked SGS validation.

## 6. Conclusions

1. The full rectangular grid and graphitic-support reporting volume give materially different ensemble summaries. The completed full-grid mean is 2.056% TGC, compared with 3.829% under fractional archive-lode weighting and 3.903% in the full-cell core.

2. The archive-derived, DEM-clipped lode envelope provides a transparent common support for reporting conditional P50 TGC, TGC spread and above-threshold occupancy. It is a sensitivity to domain representation and reporting support, not independent validation.

3. Envelope probability and spread products can be assessed for Monte Carlo stability, covariance behaviour and directional reproduction. Their interpretation remains relative to the completed model and the selected support.

4. The repeated geology-blind sensitivity retains closer selected global distribution metrics. Global fit, support alignment and geological information should therefore be reported as separate evaluation axes rather than collapsed into one model ranking.

## 7. Statements and Declarations

### 7.1 Data Availability

The collar, survey, lithology, assay, and QA/QC database belongs to the project data holder and is subject to confidentiality restrictions. Online Resource 1 (Supplementary Methods and Validation) documents the extended workflow and validation scope. Online Resource 2 (Audit-Level Metadata and Validation Workbook) provides machine-readable run configuration, variogram, validation, convergence, support-decomposition, contact, occupancy, and null-sensitivity summaries. These resources support audit of the reported calculations but cannot regenerate the proprietary project arrays. The full database may be made available to editors or reviewers for confidential examination, subject to data-owner approval.

### 7.2 Funding

This research received no specific external grant from funding agencies in the public, commercial, or not-for-profit sectors.

### 7.3 Ethics Approval

Not applicable. This geological and geostatistical study involved no human participants or animals.

### 7.4 Consent to Participate

Not applicable.

### 7.5 Consent for Publication

Not applicable.

### 7.6 Competing Interests

The author is affiliated with Sakariya Mines and Minerals Private Limited, which provided the project data used in this study. This affiliation is declared as a potential competing interest.

## 8. References

[1] Moye CD, Msabi M (2021) Mineralogical and geochemical characteristics of graphite-bearing rocks at Chenjere Area, south-eastern Tanzania: Implications for the nature and quality of graphite mineralization. Tanzan. J. Sci. 47:535-551. https://doi.org/10.4314/tjs.v47i2.11

[2] Das S, Goswami S, Chowdhury SA, De S, Das K (2026) Discovery of the world class Maramba-Tanga Graphite deposit, NE Tanzania, Africa. Ore Energy Resour. Geol. 21:100132. https://doi.org/10.1016/j.oreoa.2026.100132

[3] Case GND (2026) A time-space model of graphite mineral systems. Miner. Deposita 61:783-810. https://doi.org/10.1007/s00126-025-01412-5

[4] Deutsch CV (2023) The Place of Geostatistical Simulation through the Life Cycle of a Mineral Deposit. Minerals 13:1400. https://doi.org/10.3390/min13111400

[5] Maleki M, Emery X (2015) Joint simulation of grade and rock type in a stratabound copper deposit. Math. Geosci. 47:471-495. https://doi.org/10.1007/s11004-014-9556-8

[6] Paithankar A, Chatterjee S (2018) Grade and tonnage uncertainty analysis of an African copper deposit using multiple-point geostatistics and SGS. Nat. Resour. Res. 27:419-436. https://doi.org/10.1007/s11053-017-9364-1

[7] Fritz H, Abdelsalam M, Ali KA, Bingen B, Collins AS, Fowler AR, Ghebreab W, Hauzenberger C, Johnson PR, Kusky TM, Macey P, Muhongo S, Stern RJ, Viola G (2013) Orogen styles in the East African Orogen: A review of the Neoproterozoic to Cambrian tectonic evolution. J. Afr. Earth Sci. 86:65-106. https://doi.org/10.1016/j.jafrearsci.2013.06.004

[8] Muhongo S (1994) Neoproterozoic collision tectonics in the Mozambique Belt of East Africa: evidence from the Uluguru mountains, Tanzania. J. Afr. Earth Sci. 19:153-168. https://doi.org/10.1016/0899-5362(94)90058-2

[9] Maboko MAH (1997) P-T conditions of metamorphism in the Wami River granulite complex, central coastal Tanzania: implications for Pan-African geotectonics in the Mozambique Belt of eastern Africa. J. Afr. Earth Sci. 24:51-64. https://doi.org/10.1016/S0899-5362(97)00026-2

[10] Boniface N (2019) An overview of the Ediacaran-Cambrian orogenic events at the southern margins of the Tanzania Craton: Implication for the final assembly of Gondwana. J. Afr. Earth Sci. 150:123-130. https://doi.org/10.1016/j.jafrearsci.2018.10.015

[11] Fritz H, Tenczer V, Hauzenberger C (2023) Fold interference pattern and crustal decoupling in northern Tanzania. J. Afr. Earth Sci. 202:104940. https://doi.org/10.1016/j.jafrearsci.2023.104940

[12] Malisa EP (1998) Application of graphite as a geothermometer in hydrothermally altered metamorphic rocks of the Merelani-Lelatema area, Mozambique Belt, northeastern Tanzania. J. Afr. Earth Sci. 26:313-316. https://doi.org/10.1016/S0899-5362(98)00013-X

[13] Isaaks EH, Srivastava RM (1989) An Introduction to Applied Geostatistics. Oxford University Press, New York.

[14] Goovaerts P (1997) Geostatistics for Natural Resources Evaluation. Oxford University Press, New York.

[15] Deutsch CV, Journel AG (1998) GSLIB: Geostatistical Software Library and User's Guide, 2nd ed. Oxford University Press, New York.

[16] Chiles J-P, Delfiner P (2012) Geostatistics: Modeling Spatial Uncertainty, 2nd ed. Wiley, Hoboken, NJ.
