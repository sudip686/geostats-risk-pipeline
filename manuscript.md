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
Layer-parallel graphitic schist defines an exploration corridor, but reporting support controls whether simulated grade uncertainty is read as background dilution or lode behaviour. We evaluate a 100-realisation geology-conditioned Sequential Gaussian Simulation ensemble for a Tanzanian stratiform graphite system and restrict completed outputs to a topography-clipped archive lode envelope at 50 x 50 x 2 m reporting support. The envelope retains 55,716 fine blocks (4.313% of reporting volume), shifting mean TGC from 2.056% in the full grid to 3.829%, close to 3.921% in declustered graphitic composites. On identical envelope support, five conditioned 20-realisation subsets give median above-threshold occupancy 0.656 and TGC spread 3.199%, compared with 0.595 and 5.068% across five geology-blind families; the null families retain closer distribution fit. Probability and spread fields stabilise strongly by 75 realisations. Persistent occupancy lies nearer sampled composites, whereas high-spread columns occur farther from support and more often on envelope edges. Geological conditioning therefore converts grade uncertainty into support, persistence and spread diagnostics for relative geological follow-up in layered industrial minerals.

**Keywords:** graphite; conditional simulation; reporting support; geological uncertainty; exploration evaluation; Tanzania

## 1. Introduction

Graphite-bearing metasedimentary horizons in the Tanzanian Mozambique Belt commonly follow compositional layering and metamorphic fabric. That layer-parallel continuity defines an exploration target, but it does not make grade, contact position, weathering state or package geometry equally continuous between drillholes.

Published studies establish the regional setting, host rocks and mineralogical character of Tanzanian graphite occurrences [1, 2, 3]. The unresolved mining-geology problem is how to distinguish uncertainty in graphitic support from uncertainty in grade when a simulation grid contains both lode and background volume.

Conditional simulation provides multiple spatial outcomes through which that distinction can be tested [4]. Studies of stratabound copper and African mineral deposits show that domain representation, anisotropy and reporting support can materially alter apparent uncertainty behaviour [5, 6], yet this separation has rarely been quantified for stratiform graphite.

This study asks: (1) how strongly reporting support changes ensemble grade summaries; (2) where conditional grade spread and above-threshold persistence occur inside an interpreted graphitic envelope; (3) what remains different when geology-conditioned and geology-blind ensembles are evaluated inside exactly the same volume; and (4) which diagnostics can guide relative geological follow-up. The contribution is a geology-led framework that evaluates support alignment, global distribution fit and geological information as distinct evidence axes.

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

### 3.1 Drillhole Database and Analytical and Workflow Quality Assurance and Quality Control (QA/QC)

The study uses 100 drillholes, 3,350 assay intervals and 1,248 lithology records. Table 1 follows the data from raw assays and lithology logs through desurveying, compositing, domain assignment and Online Resource 2. Before modelling, workflow checks covered interval validity, assay/lithology support, survey availability and the 100-hole study policy; four surveyed holes with incomplete assay/lithology support were excluded from study metrics.

For the 2024-2025 drilling campaign represented by the curated database, project analytical QA/QC records document preparation at SGS Mwanza by drying, crushing to less than 2 mm, splitting and pulverising to 85% passing 75 micrometres, followed by infrared-combustion TGC analysis. Control insertion comprised 93 certified reference materials, 94 blanks, 93 coarse duplicates and 93 pulp duplicates (373 controls; 11.1% of submissions). Batch review reported blanks below 0.05% TGC, no CRM action-limit failures and duplicate correlation above 0.98 at the stated precision criterion. These records establish the analytical suitability of the assay population used here; the reproducible workflow separately audits data transfer and modelling calculations.

### 3.2 Compositing and Support

Assay intervals were desurveyed and composited to nominal 2 m bins within successive lithological groups. Composite TGC was calculated by length weighting the assay overlap:

Equation (1):

```math
Z_{\mathrm{comp}} = \frac{\sum_i L_i Z_i}{\sum_i L_i}
```

where each contributing assay is weighted by its sampled overlap. A minimum retained nominal bin length of 0.5 m was used at group edges. Raw-interval reconciliation found 88 partly assay-covered composites (2.13% of 4,129). Internal unsampled portions total 79.42 m, or 0.998% of the 7957.70 m nominal span; 32 bins are less than half assay-covered. Excluding all partly covered bins from the descriptive length-weighted mean changes it from 4.146% to 4.156% TGC. The completed SGS retains the archived composite set, so this audit constrains global support sensitivity but does not quantify local simulation influence.

Simulation used 25 m x 25 m x 2 m cells. The lateral dimensions retain plan-view graphitic-body morphology, while the 2 m vertical dimension follows the nominal composite and logged-contact scale. Results were aggregated to 50 m x 50 m x 2 m reporting support. The two-by-two lateral aggregation is tied to local drill spacing and stabilises map-scale probability and spread diagnostics while retaining vertical resolution.

No top-cut was applied because the graphitic 2 m population does not contain a detached high-grade tail. It has n = 3,948, mean 4.27% TGC, median 3.96% TGC, maximum 14.67% TGC and COV 0.53. A 99.5th-percentile cap at 11.99% TGC would affect 19 composites (0.48%) and change the mean and variance by only -0.13% and -1.88%, respectively.

### 3.3 Domain Definition

Composites were grouped into fresh graphitic, weathered graphitic and host/waste categories. The 3% TGC threshold was selected from the study dataset before SGS as a geological screening threshold separating weakly graphitic/background material from more continuous graphitic-schist support in the local histogram and logs. In the canonical 2 m composites it lies between the lower quartile (2.358% TGC) and median (3.849% TGC), and 64.26% of composite metres are at or above it. The threshold is used only for domain checks and above-threshold model occupancy; no economic assumptions are applied. Table 3 records the threshold, boundary, search-neighbourhood and variogram settings. The hard-boundary case is a geological-prior end member that prevents grade conditioning across fresh graphitic, weathered graphitic and host/waste categories within each realisation.

### 3.4 Categorical Sensitivity Used by Grade SGS

Fresh graphitic, weathered graphitic and host/waste classes were assigned from logged composites and sampled from fixed local inverse-distance probability scores within the configured anisotropic search. The archived implementation draws categories independently at grid nodes rather than using indicator SGS, transition probabilities or a spatially coherent body model. Grade SGS was then performed within the sampled class structure. Raw class frequencies and entropy are retained in Online Resource 2 as secondary sensitivity diagnostics; the archive-derived lode envelope provides the primary reporting support.

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

### 3.9 Archive-Derived Lode Envelope and Spatial Diagnostics

The archive source contains seven lode identifiers. Exact centre matching, common-footprint screening and clipping below its DEM-derived surface retain six identifiers and 55,716 blocks; L01 contributes 51,809 blocks (92.99%). The available envelope was generated algorithmically from graphitic-coded 2 m composites, a 3% TGC screening rule, gap rules, spatial clustering and roof-and-floor interpolation. The analysis reads only x, y, z, block dimensions, lode identity and topography. Estimated TGC, kriging variance, density, classification, tonnes and contained graphite are excluded.

The 25 x 25 x 2 m archive blocks were mapped by exact centre alignment to the canonical EPSG:32737 grid. At reporting support, each 50 x 50 x 2 m cell receives a fractional lode weight f equal to its retained fine-block count divided by four. For each realisation, the weighted mean is the sum of cell TGC multiplied by f divided by the sum of f. Any-intersection and full-cell-core summaries provide sensitivity brackets around the fractional-volume result. Vertical envelope occupancy is the sum of retained 2 m intervals in each plan column; it is not treated as true thickness.

Plan-map patterns were quantified before interpretation. High spread is the upper decile of envelope-weighted plan P90-P10 TGC spread. Persistent above-threshold occupancy is plan P(TGC > 3%) greater than or equal to 0.80. Reporting-column centres were related to the nearest sampled composite in plan projection, and footprint-edge columns were identified by a one-cell, eight-neighbour erosion. These are support diagnostics rather than prediction errors.

### 3.10 Geology-Blind Composite Null and Matched-Envelope Comparison

Five independent no-domain isotropic families were completed with seeds 9101, 9201, 9301, 9401 and 9501, each containing 20 realisations. The null configuration uses direct 50 x 50 x 2 m simulation, one whole-population normal-score transform, isotropic 150 m covariance and search, 8-24 neighbours and an enabled vertical trend. The canonical configuration uses 25 x 25 x 2 m simulation aggregated to the same reporting support, stochastic hard domains, domain-wise transforms, 250/200/20 m geological-axis covariance and search, 3-20 neighbours and no grade trend. This is a composite configuration sensitivity, not a one-factor ablation.

For a realisation-count and volume-matched comparison, the same fractional archive weight was applied to every null family and to five contiguous, non-overlapping 20-realisation subsets of the canonical ensemble. Envelope means, P(TGC > 3%), cellwise P90-P10 spread, decluster-weighted graphitic histogram overlap, weighted Q-Q RMSE and geological-axis swath correlations were recomputed inside the identical support. All five seed families and all five canonical subsets are reported without performance selection.

### 3.11 Withheld-Composite Validation Baseline



The categorical probabilities were validated independently of grade SGS by five-fold grouped cross-validation (CV). Complete drillholes were withheld, the fixed local-probability algorithm was recomputed from the remaining holes only, and predictions were evaluated at every withheld composite. Multiclass performance is reported using macro-F1, balanced accuracy, log loss and a three-class confusion matrix. Fresh plus weathered graphitic classes were combined against host/waste to calculate ROC-AUC, Brier score, Brier skill relative to each fold's training prevalence and ten fixed-width reliability bins. Predictions were separated into locations with at least one retained-hole composite inside the configured anisotropic search and locations invoking the deterministic host/waste no-support fallback. Normalised Shannon entropy was tested as a relative error-ranking score using ROC-AUC against the held-out class-error indicator. A leakage-free calibration sensitivity used four-fold grouped inner predictions to fit a logistic mapping within each outer training fold before application to its withheld holes; this mapping was not applied to the archived categorical realisations. Fold construction uses seed 20260707 and requires zero drillhole overlap.

Directional swaths were computed in the configured strike, down-dip and thickness-normal coordinate system. For each of ten equal-width bins, observed composite TGC is shown only when at least five composites are present. Each realisation was averaged over reporting cells in the same bin, and the plotted P10, P50 and P90 curves are percentiles across those 100 realisation-level bin means. Aligned bars beneath each swath report the observed composite count, and a dashed line marks the five-composite display threshold. Separating sample support from the grade curves allows directional agreement, envelope width and data density to be read together without annotation obscuring the observations.

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

Raw-interval reconciliation identifies 88 partly assay-covered composites containing 79.42 m of internal unsampled span (0.998% of nominal composite metres). Removing these bins from the descriptive length-weighted mean changes TGC by 0.009 percentage points.

### 4.2 Structural and Variogram Evidence

Directional continuity is anisotropic, but the empirical resolution differs among axes. The along-strike fit reaches 1187.5 m, beyond the 500 m experimental window, and is therefore reported as greater than 500 m and not sill-constrained. Down-dip and thickness-normal provisional range proxies are 90.7 m and 21.9 m, respectively. Figure 2 reports the observed corridor, geological axes and the regularised 250/200/20 m search model used by SGS; Table 3 lists the corresponding settings.

### 4.3 Reporting Support, Ensemble Behaviour and Matched Null Comparison

After common-footprint and topography checks, the retained envelope occupies 4.313% of reporting volume, intersects 19,286 reporting cells and contains 8,938 full-cell lode-core cells. The full-grid, any-intersection, fractional-volume and full-cell-core means are 2.056%, 3.783%, 3.829% and 3.903% TGC, respectively. The archive source has seven lode identifiers, but retained blocks are strongly concentrated in L01: 51,809 of 55,716 blocks (92.99%). L02 lies outside the common SGS footprint.

Inside identical fractional envelope support, the five canonical 20-realisation subsets have median mean TGC 3.825%, P(TGC > 3%) 0.656, P90-P10 spread 3.199%, histogram overlap 0.876 and Q-Q RMSE 0.474% TGC. The five null families give 3.892%, 0.595, 5.068%, 0.925 and 0.233%, respectively. Median strike/down-dip/thickness-normal swath correlations are 0.761/0.822/0.834 for the canonical subsets and 0.734/0.857/0.856 for the null families.

At n = 75, envelope probability MAE is 0.018, probability correlation is 0.995, and spread correlation is 0.942 relative to the 100-realisation reference. Matched-space variogram reproduction has weighted RMSE 0.237; the thickness-normal direction retains two pair-supported lags. Figure 7 and Table 4 report these results.

### 4.4 Population and Physical-Domain Checks

The physical-domain audit found no negative reporting-support values in the canonical trend-disabled ensemble: the minimum is 0.000% TGC and the negative-cell proportion is 0.00%. Replacing negative values with zero therefore leaves the mean, P10, P50, P90 and 3% occupancy unchanged.

### 4.5 Contact and Weathering Controls

The corrected weathering comparison is restricted to graphitic-domain composites. Weathered graphitic intervals (n = 382) average 4.80% TGC, compared with 4.21% TGC for fresh graphitic intervals (n = 3566). The mean difference is 0.59 percentage points (95% CI 0.32 to 0.86; Hedges g = 0.26; Welch p < 0.001). The hole-cluster bootstrap interval is 0.11 to 1.06 percentage points, whereas the 79-hole paired comparison is inconclusive (Wilcoxon p = 0.167).

The signed graphitic-host profile contains 711 composites around 134 contiguous transitions in 42 drillholes. Graphitic-side mean TGC exceeds host/waste-side mean TGC by 2.42 percentage points. The separate unsigned graphitic-only distance-bin comparison is nonsignificant (ANOVA p = 0.467; Kruskal-Wallis p = 0.496; Levene p = 0.747). Figure 4a displays the signed profile and hole-cluster intervals.

Figure 4b displays the fresh and weathered graphitic TGC distributions. Figure 4c replots the published XRF weathering data of Das et al. [2] as contextual evidence and is not a new project measurement or an SGS validation result.

### 4.6 Envelope-Constrained Spatial Uncertainty Products

The plan footprint contains 1,254 envelope-intersecting reporting columns. The upper-decile spread threshold is 4.041% TGC, selecting 126 columns (10.05%); 55 of these (43.7%) lie in the central northing third. Their median plan distance to the nearest sampled composite is 90.3 m, compared with 78.4 m for all envelope columns. High-spread columns occur beyond 100 m from a sampled composite in 41.3% of cases and on the plan-footprint edge in 32.5% of cases; the corresponding background proportions are 31.2% and 22.9%.

Persistent plan occupancy, defined by P(TGC > 3%) greater than or equal to 0.80, occurs in 197 columns (15.71%). These columns have median nearest-composite distance 40.4 m and mean vertical envelope occupancy 37.8 m, compared with 22.2 m across the footprint. High spread and persistent occupancy coincide in 20 columns (1.59%); 55.0% of that joint set lies in the northern third. Figure 5 maps these plan patterns. Figure 6 shows their expression on the selected east-west section together with fixed realisations 1, 50 and 100 on a common scale.

### 4.7 Categorical and Withheld Validation Results

Five-fold hole-grouped categorical validation gives macro-F1 0.356, balanced accuracy 0.442 and graphitic-host ROC-AUC 0.708. Within anisotropic search support, Brier skill is -0.407 and entropy ranks held-out classification errors with ROC-AUC 0.650. The full confusion matrix and reliability bins are supplied in Online Resource 2. The 500 m block, leave-hole and leave-section grade baselines have RMSE 2.261%, 2.179% and 2.232% TGC, respectively. Table 4 assembles the validation results on their corresponding evidence axes.

## 5. Discussion

### 5.1 Geological Support and Reporting-Envelope Effects

The central geological result is the separation of computational support from graphitic support. The full grid mixes the interpreted lode corridor with a large background volume, whereas the fractional envelope asks how the completed ensemble behaves where graphitic support has already been interpreted. The shift from 2.056% to 3.829% TGC therefore demonstrates why full-grid and lode-support means answer different volume questions; the similarity to graphitic-composite grade is a support decomposition, not independent validation.

This distinction follows the broader geostatistical principle that uncertainty is inseparable from domain definition and support. Simulation carries uncertainty through alternative spatial outcomes, while geological domains determine which outcomes are compared and reported [4, 5]. Paithankar and Chatterjee [6] similarly show in an African mineral-deposit setting that ensemble behaviour must be read together with spatial support rather than from global reproduction alone. For the present graphite system, the envelope makes that support choice explicit and auditable.

### 5.2 Spatial Meaning and Geological Follow-Up

The plan patterns distinguish two practical situations. Persistent occupancy is concentrated closer to sampled composites and in thicker vertically occupied columns, so it identifies parts of the interpreted corridor where above-threshold support recurs across the ensemble. High spread is farther from sampled support and disproportionately represented on footprint edges, pointing to locations where contact position, continuation of the package, or local grade variability remains less constrained. The small joint set combines persistence with broad conditional spread; those columns are the strongest candidates for section review, contact verification and holes oriented across the package.

The transferable value is the separation of geometry and grade uncertainty. In layered industrial-mineral and stratabound systems, a coherent host horizon can coexist with uncertain margins and grade distribution. Ensemble geological studies likewise use spatial variability and topology to locate where sparse data and structural assumptions leave geometry uncertain [17, 18, 19, 20]. Joint rock-type/grade simulation and African deposit studies reach the complementary conclusion that domain architecture and grade uncertainty should be carried together but diagnosed separately [5, 6]. Here, Figures 5 and 6 turn those principles into mappable follow-up classes rather than one undifferentiated uncertainty surface.

### 5.3 What the Matched Null Comparison Resolves

Applying both model families to the identical envelope shows that the null's closer distribution fit is not produced only by background volume: its median envelope histogram overlap remains 0.925 versus 0.876. At the same time, the conditioned subsets show higher above-threshold persistence and a 36.9% narrower median spread, with slightly stronger strike correlation; the null has stronger median down-dip and thickness-normal correlations. Repetition across all five seeds establishes the robustness of this global-fit behaviour. The comparison therefore supports two explicit evaluation axes: distribution reproduction and the geological organisation of conditional uncertainty [4, 21].

### 5.4 Contact, Weathering and Categorical Information

Figure 4 establishes a marked grade contrast across logged graphitic-host transitions, which supports treating contact position as an explicit uncertainty axis. The modest fresh-weathered mean contrast is not reproduced consistently by paired-hole or three-class validation, so weathering is retained as a secondary grouping variable rather than the main geological control. This evidence connects directly to Figures 5 and 6: high spread concentrated farther from drilling and along envelope edges identifies where contact position and package continuation require geological verification.

Hole-grouped validation shows that the local categorical scorer retains modest graphitic-host ranking, while fresh and weathered graphite remain poorly separated. Raw categorical probabilities and entropy are consequently most useful for relative within-support patterns. This hierarchy is consistent with joint domain-grade simulation practice, where categorical architecture must be evaluated independently of grade reproduction [5, 22, 23, 24]. Boundary-aware studies likewise show that contact behaviour should be diagnosed rather than assumed [25, 26]. Plurigaussian simulation provides a more spatially coherent alternative for future categorical-domain modelling than independent local draws [27].

### 5.5 Implications for Graphite Exploration and Resource Evaluation

Table 5 translates each output into a specific geological action. The envelope defines the volume being evaluated; persistent occupancy identifies recurrent graphitic support; high spread marks conditional grade uncertainty; and their overlap prioritises places where support persists but its grade range remains broad. Raw categorical entropy is retained for re-logging and contact review within mapped search support, while the matched null comparison prevents model selection from being reduced to histogram agreement. Comparable mining-uncertainty frameworks tie uncertainty classes to investigation priorities and decision use [28, 29].

Used together, the products establish an efficient follow-up sequence: review edge and high-spread columns on section, check whether mapped contacts and foliation support the envelope geometry, then place cross-package or infill drilling where the expected information gain is greatest. This is directly relevant to layered industrial minerals because it separates the question "is the host package present?" from "how variable is grade within it?" before either is converted into a mine-planning assumption.

### 5.6 Limitations and Future Validation

The evidence is dominated by L01, which supplies 92.99% of retained archive blocks, so transfer among the six retained lode identifiers remains to be tested. The envelope shares drillhole, lithology and threshold information with the SGS and uses simplified vertical-run geometry; the archived SGS retains 88 partly assay-covered composites representing 0.998% of nominal composite metres; categorical classes are sampled by independent local draws; the null families change several controls together; and withheld grade baselines are not blocked reruns of the final SGS. Future validation should therefore test corrected composite support, desurveyed section interpretations, a calibrated plurigaussian or rapid-updating domain model [27, 30], locally varying structure and independent drilling or fully blocked SGS calibration.

## 6. Conclusions

1. Reporting support controls the geological meaning of ensemble statistics. Mean TGC is 2.056% over the full grid and 3.829% under fractional lode-envelope weighting, close to the 3.921% declustered graphitic-composite mean.

2. The conditioned ensemble resolves persistent above-threshold support from broad conditional grade spread. Persistent columns lie closer to sampled composites; high-spread columns lie farther from support and occur more often along envelope edges.

3. On identical support and equal realisation count, the null families retain closer graphitic-distribution fit, while the conditioned subsets have higher above-threshold persistence, narrower TGC spread and slightly stronger strike reproduction. Global fit and geological information are therefore complementary evaluation axes.

4. The practical workflow is transferable to layered industrial minerals: define common geological support, compare model families inside that support, test ensemble and directional behaviour, and target follow-up where support persistence and grade spread overlap.

## 7. Statements and Declarations

### 7.1 Data Availability Statement

The collar, survey, lithology, assay, and QA/QC database used in this study belongs to the project data holder and is subject to confidentiality restrictions; it is not publicly available. The data that support the findings of this study are available from the corresponding author upon reasonable request, subject to data-owner approval. Online Resource 1 (Supplementary Methods and Validation) documents the extended workflow and validation scope. Online Resource 2 (Audit-Level Metadata and Validation Workbook) provides machine-readable run configuration, variogram, validation, convergence, support-decomposition, contact, occupancy, and null-sensitivity summaries. These supplementary resources support audit of the reported calculations but cannot regenerate the proprietary project arrays. The confidential project database may be made available to editors or reviewers for confidential examination, subject to data-owner approval.

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

[17] Lindsay M, Ailleres L, Jessell M, De Kemp EA, Betts PG (2012) Locating and quantifying geological uncertainty in three-dimensional models: Analysis of the Gippsland Basin, southeastern Australia. Tectonophysics 546-547:10-27. https://doi.org/10.1016/j.tecto.2012.04.007

[18] Schaaf A, Bond CE (2019) Quantification of uncertainty in 3-D seismic interpretation: Implications for deterministic and stochastic geomodeling and machine learning. Solid Earth 10:1049-1061. https://doi.org/10.5194/se-10-1049-2019

[19] Schaaf A, de la Varga M, Wellmann F, Bond CE (2021) Constraining stochastic 3-D structural geological models with topology information using approximate Bayesian computation in GemPy 2.1. Geosci. Model Dev. 14:3899-3913. https://doi.org/10.5194/gmd-14-3899-2021

[20] Nie X, Lu C, Luo K (2023) Uncertainty assessment of 3D geological models based on spatial diffusion and merging model. Open Geosci. 15:20220456. https://doi.org/10.1515/geo-2022-0456

[21] Bassani MAA, Costa JFCL, Deutsch CV (2024) A comparative study between the direct and indirect methods in geostatistical simulation. Min. Metall. Explor. 41:3669-3691. https://doi.org/10.1007/s42461-024-01087-y

[22] Talebi H, Hosseinzadeh Sabeti E, Azadi M, Emery X (2016) Risk quantification with combined use of lithological and grade simulations: Application to a porphyry copper deposit. Ore Geol. Rev. 75:42-51. https://doi.org/10.1016/j.oregeorev.2015.12.007

[23] Mery N, Emery X, Caceres A, Ribeiro D, Cunha E (2017) Geostatistical modeling of the geological uncertainty in an iron ore deposit. Ore Geol. Rev. 88:336-351. https://doi.org/10.1016/j.oregeorev.2017.05.011

[24] Iliyas N, Madani N (2021) An enhanced co-simulation technique for resource modelling using grade domaining: A case study from an iron ore deposit. Appl. Earth Sci. 130:81-106. https://doi.org/10.1080/25726838.2021.1882644

[25] Emery X, Maleki M (2019) Geostatistics in the presence of geological boundaries: Application to mineral resources modeling. Ore Geol. Rev. 114:103124. https://doi.org/10.1016/j.oregeorev.2019.103124

[26] Maleki M, Emery X (2020) Geostatistics in the presence of geological boundaries: Exploratory tools for contact analysis. Ore Geol. Rev. 120:103397. https://doi.org/10.1016/j.oregeorev.2020.103397

[27] Emery X (2007) Simulation of geological domains using the plurigaussian model: New developments and computer programs. Comput. Geosci. 33:1189-1201. https://doi.org/10.1016/j.cageo.2007.01.006

[28] Tichauer R, De Tomi G (2019) The Tichauer-DeTomi Matrix: A tool for assessment of geological uncertainty in small-scale mining. Min. Metall. Explor. 36:579-588. https://doi.org/10.1007/s42461-019-0052-z

[29] Lindi OT, Aladejare AE, Ozoji TM, Ranta J-P (2024) Uncertainty Quantification in Mineral Resource Estimation. Nat. Resour. Res. 33:2503-2526. https://doi.org/10.1007/s11053-024-10394-6

[30] Abulkhair S, Dowd PA, Xu C (2026) Pluri-Gaussian rapid updating of geological domains. Math. Geosci. https://doi.org/10.1007/s11004-025-10261-x
