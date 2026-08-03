# CEMAT-Stack V4

**Clinical Expert and Modality-Aware Stacking for leakage-safe pediatric appendicitis severity prediction.**

CEMAT-Stack V4 is the recommended primary pipeline for the MAT-Appendix project. It uses the complete prognostic cohort of 463 confirmed appendicitis patients (118 complicated, 345 uncomplicated) and does not require raw ultrasound images. The previous CGR-MAT image pipeline remains an exploratory secondary analysis.

## Why this pipeline

The raw-image cohort is small and incomplete. V4 therefore prioritizes a data-efficient structured-data design with strict nested validation:

1. Global Logistic Regression, ExtraTrees, HistGradientBoosting, XGBoost and CatBoost models.
2. Clinical/laboratory, ultrasound-tabular and missingness-pattern experts.
3. Inner-fold out-of-fold predictions for every level-1 expert.
4. Sparse non-negative global and expert blends.
5. Availability-aware CEMAT meta-stacking using expert disagreement and modality availability.
6. Cross-fitted Platt calibration.
7. Threshold selection using outer-training OOF predictions only.
8. Five repeats of stratified five-fold outer validation.

## Leakage exclusions

The code rejects identifiers, outcome labels, management and post-decision/direct complication variables, including:

- `Diagnosis`, `Severity`, `Management`, `Length_of_Stay`, `US_Number`;
- `Peritonitis`, `Perforation`, `Appendicular_Abscess`, `Abscess_Location`;
- operation, surgery, pathology, histology, postoperative and discharge variables.

The run aborts unless the cohort signature is exactly `N=463`, `complicated=118`, `uncomplicated=345`.

## Models reported

- `CatBoost`: strong single-model baseline.
- `GlobalBlend`: constrained blend of five complete-feature models.
- `ExpertBlend`: constrained blend of global and modality experts.
- `CEMATStack`: availability-aware calibrated meta-stack.

Two operating points are exported:

- `balanced`: prioritizes balanced accuracy, F1 and MCC while preferring sensitivity ≥0.82 and specificity ≥0.80.
- `high_sensitivity`: targets sensitivity ≥0.90 and then maximizes specificity.

## Run in Colab

Open `CEMAT_Stack_V4_Colab.ipynb` and run all cells. A GPU is not required because this is a structured-data pipeline. Google Drive is used for checkpoints and outputs.

Default scientific configuration:

- 5 outer folds × 5 repeats = 25 outer evaluations;
- 3 inner folds for level-1 predictions;
- 3 meta folds;
- 3 calibration folds;
- 3,000 stratified paired bootstrap replicates.

For a debugging run only, set before running the loader:

```python
import os
os.environ["CEMAT_REPEATS"] = "1"
os.environ["CEMAT_BOOTSTRAPS"] = "100"
```

Do not report debugging metrics in a paper.

## Resume behavior

Every outer fold is saved independently. Rerunning the notebook loads completed fold checkpoints unless `CEMAT_FORCE_RESTART=1` is set.

## Main outputs

The run directory is written to:

```text
MyDrive/MAT-Appendix/cemat_v4_runs/cemat_stack_v4_<config-hash>/
```

Important artifacts:

- `config.json`, `environment_freeze.txt`;
- `cohort_summary.csv`, `leakage_audit.csv`, `fallback_audit.csv`;
- fold predictions and fold logs for all 25 evaluations;
- `all_repeated_nested_predictions.csv`;
- `repeat_metrics.csv` and `summary_mean_std.csv` — primary paper evidence;
- `per_patient_consensus.csv` and `consensus_metrics.csv` — secondary summary;
- `paired_bootstrap.csv`;
- `final_decision.json` and `publication_audit.json`;
- ROC, PR, calibration and metric-comparison PNG figures;
- `manifest_sha256.csv` and a ZIP archive.

## Interpretation

Do not require every metric to exceed 0.93. A credible result should be judged using discrimination, class-balanced performance, probability quality and uncertainty together. The code records target checks for ROC-AUC, accuracy, sensitivity, specificity, F1 and Brier score, but it never alters the validation protocol to force those targets.

Use repeat-level mean ± standard deviation as the primary result. Consensus metrics and paired bootstrap comparisons are secondary. Do not claim that CEMAT is superior unless its paired uncertainty supports the claim without material Brier-score degradation.

## Current scope

This is retrospective, single-center research software. It is not a clinical decision system. External validation remains necessary for a strong journal-level deployment claim.
