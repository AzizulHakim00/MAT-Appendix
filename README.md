# MAT-Appendix

**Missingness-Aware Feature-Token Transformer for leakage-safe pediatric appendicitis severity prediction.**

## Primary experiment

Open `notebooks/MAT_Appendix_Primary_5Fold.ipynb` in Google Colab and run its single code cell. The notebook:

- fetches UCI dataset 938 directly; no local dataset upload is required;
- forms the complicated-vs-uncomplicated appendicitis cohort;
- removes post-decision, procedure/pathology, target and direct complication-definition variables;
- fits MAT-Appendix exactly once in each of five outer folds;
- creates genuine out-of-fold predictions;
- performs cross-fitted Platt calibration and cross-fitted threshold selection;
- calculates ECE, MCE, Brier score, log loss and 1,000-replicate bootstrap CIs;
- creates ROC, PR, calibration, confusion matrix, missingness and training figures;
- produces fold-aggregated permutation XAI, representative-fold SHAP and LIME;
- writes a compressed reproducibility PKL with five CPU model state dictionaries;
- saves tables, 300-DPI PNG/PDF figures, JSON results, a model card and a ZIP;
- optionally commits the complete run under `results/runs/<RUN_ID>/`.

## GitHub auto-save from Colab

Create a fine-grained GitHub token restricted to this repository with **Contents: Read and write**. In Colab, open **Secrets**, add `GITHUB_TOKEN`, and enable notebook access. The token is read securely, never printed, and never written to an artifact.

## Reusing results

Run `notebooks/MAT_Appendix_Load_Artifacts.ipynb`. It reads `results/LATEST_RUN.txt`, downloads the corresponding PKL and exposes:

- fixed fold indices for fair future baselines;
- fold-specific preprocessing dictionaries;
- five MAT-Appendix state dictionaries;
- raw and calibrated OOF predictions;
- calibration objects and thresholds;
- metrics, confidence intervals and XAI arrays.

Future baseline, ablation or robustness notebooks should reuse the saved folds and compare against the saved MAT-Appendix OOF predictions. The proposed model does not need to be trained again.

## Dataset

Regensburg Pediatric Appendicitis, UCI dataset ID 938 / Zenodo DOI `10.5281/zenodo.7711412`.

## Important scientific scope

This is retrospective research code, not a clinical decision system. The primary endpoint is severity among records labeled complicated or uncomplicated appendicitis. The notebook intentionally excludes direct target-defining and post-decision features.
