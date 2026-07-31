# MAT-Appendix Conference-Complete Run Guide

## Primary notebook

Open `MAT_Appendix_Conference_Complete.ipynb`, select **Runtime > Change runtime type > T4 GPU**, and run the single code cell.

## Enable GitHub export

Create a fine-grained GitHub token restricted to `AzizulHakim00/MAT-Appendix` with **Contents: Read and write**. In Colab Secrets add `GITHUB_TOKEN` and enable notebook access.

Without the secret, all outputs still save to Google Drive, but GitHub export is skipped.

## Corrected study definition

- Cohort: confirmed appendicitis only (`Diagnosis == appendicitis`)
- Target: complicated versus uncomplicated appendicitis
- Evaluation: identical stratified 5-fold out-of-fold testing
- Proposed MAT-Appendix training: exactly five fits

## Models

Five classical baselines: Logistic Regression, Random Forest, XGBoost, CatBoost and HistGradientBoosting.

Additional comparisons: ordinary FT-Transformer, MAT without learnable missingness embedding, and proposed MAT-Appendix.

## Generated outputs

The notebook reports accuracy, error rate, balanced accuracy, precision/PPV, recall/sensitivity/TPR, specificity/TNR, NPV, F1, F2, MCC, Cohen's kappa, ROC-AUC, PR-AUC, Brier, log loss, ECE, MCE, FPR, FNR and TP/TN/FP/FN.

It also creates 1,000-replicate confidence intervals, cross-fitted calibration, 90% and 95% sensitivity operating points, paired bootstrap tests, model-specific confusion matrices, error/calibration/probability-loss plots, SHAP, LIME, permutation importance, and age/sex/missingness subgroup analysis.

## Output paths

Drive: `MyDrive/MAT-Appendix/runs/<RUN_ID>/`

GitHub: `results/runs/<RUN_ID>/` and `results/LATEST_RUN.txt`

## Uploading a completed Drive run without retraining

Use `MAT_Appendix_Upload_Existing_Conference_Run.ipynb` after adding `GITHUB_TOKEN`.

## Loading results later

Use `MAT_Appendix_Load_Conference_Artifacts.ipynb`. It loads the latest PKL without retraining any model.
