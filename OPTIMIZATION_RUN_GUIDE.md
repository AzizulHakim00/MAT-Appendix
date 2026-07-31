# MAT-Appendix Nested Optimization Run Guide

## Scientific purpose

This notebook evaluates whether leakage-safe hyperparameter tuning, admission-time feature engineering, and stacking can improve the corrected confirmed-appendicitis severity endpoint.

The requested values—ROC-AUC 0.95+, sensitivity 0.95+, specificity 0.95+, and accuracy 0.97+—are goals only. The pipeline does not alter outer-test labels, tune on outer-test predictions, or select target-derived variables to force those values.

## Open the notebook

[Open MAT_Appendix_Optimization_Primary.ipynb in Colab](https://colab.research.google.com/github/AzizulHakim00/MAT-Appendix/blob/main/MAT_Appendix_Optimization_Primary.ipynb)

## Before running

1. In Colab, choose **Runtime → Change runtime type → T4 GPU**.
2. Save a copy in Google Drive.
3. Add a fine-grained GitHub token to Colab Secrets as `GITHUB_TOKEN` and enable notebook access. The token needs **Contents: Read and write** for this repository.
4. Run the notebook's single code cell.

## Evaluation design

- Cohort: `Diagnosis == appendicitis`.
- Target: complicated versus uncomplicated appendicitis.
- Five untouched outer test folds.
- Three-fold inner cross-validation for hyperparameter selection, calibration, meta-model selection, and operating thresholds.
- Deterministic admission-time feature engineering only.
- Strict removal of identifiers, target variables, post-decision variables, pathology/procedure variables, and direct complication-defining fields.

## Models

1. Tuned Logistic Regression
2. Tuned Random Forest
3. Tuned Extra Trees
4. Tuned XGBoost
5. Tuned CatBoost
6. Nested MAT-Appendix
7. Leakage-safe optimized stacked ensemble

The stacked ensemble is selected only from outer-training data. Its candidate meta-learners include regularized logistic stacking, shallow histogram gradient boosting, and a nonnegative convex probability blender.

## Important runtime note

Leakage-safe stacking requires MAT-Appendix predictions inside every outer-training fold. Therefore MAT is trained in three inner folds plus one outer-training final fit for each of five outer folds (20 fits total). This is heavier than the earlier five-fit primary notebook but is required for a valid nested ensemble comparison.

## Generated outputs

The run saves to:

```text
MyDrive/MAT-Appendix/optimization_runs/<RUN_ID>/
```

Main files:

```text
mat_appendix_optimized_reproducibility.pkl
paper_results_optimized.json
software_versions.json
LIMITATIONS.md
manifest.json
tables/metrics.csv
tables/fold_metrics.csv
tables/selected_hyperparameters.csv
tables/meta_selection.csv
tables/operating_points_training_only.csv
tables/target_gap.csv
tables/paired_bootstrap.csv
tables/all_nested_oof_predictions.csv
figures/*.png
figures/*.pdf
```

When GitHub export succeeds, files are copied to:

```text
results/optimization_runs/<RUN_ID>/
results/LATEST_OPTIMIZATION_RUN.txt
```

## What to inspect first

1. `metrics.csv`: compare the optimized stacked ensemble with Random Forest, Extra Trees, CatBoost, XGBoost, Logistic Regression, and MAT-Appendix.
2. `target_gap.csv`: shows exactly how far each metric is from the requested target.
3. `meta_selection.csv`: shows which meta-model was selected in each outer fold.
4. `paired_bootstrap.csv`: determines whether the ensemble improvement is statistically supported.
5. `operating_points_training_only.csv`: reports 90%/95% sensitivity, 95% specificity, and whether simultaneous 95% sensitivity/specificity was feasible inside training folds.
6. `all_nested_oof_predictions.csv`: preserves all probabilities/classes for future plots without retraining.

## Interpretation rule

A model is not considered improved merely because one threshold produces high sensitivity. A strong result should show:

- higher PR-AUC and ROC-AUC than the strongest baseline;
- acceptable sensitivity and specificity at the same leakage-safe threshold;
- lower or comparable Brier score and calibration error;
- paired-bootstrap confidence intervals supporting the gain;
- stable performance across outer folds.

If near-perfect results appear suddenly, re-audit leakage, circular target definitions, post-decision variables, and duplicate patients before writing the paper.

## Load results later without retraining

[Open MAT_Appendix_Load_Optimization_Artifacts.ipynb in Colab](https://colab.research.google.com/github/AzizulHakim00/MAT-Appendix/blob/main/MAT_Appendix_Load_Optimization_Artifacts.ipynb)
