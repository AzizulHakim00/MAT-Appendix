# MAT-Appendix

**Missingness-Aware Feature-Token Transformer for leakage-safe pediatric appendicitis severity prediction.**

[![Open Primary Notebook in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AzizulHakim00/MAT-Appendix/blob/main/MAT_Appendix_Primary_5Fold.ipynb)

[![Open Artifact Loader in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AzizulHakim00/MAT-Appendix/blob/main/MAT_Appendix_Load_Artifacts.ipynb)

## Primary experiment

Open `MAT_Appendix_Primary_5Fold.ipynb` and run its single code cell. The notebook downloads and executes the versioned source fragments from this repository, so all tables and figures still appear directly in the same Colab cell.

The pipeline:

- fetches UCI dataset 938 directly; no local dataset upload is required;
- forms the complicated-vs-uncomplicated appendicitis severity cohort;
- removes post-decision, procedure/pathology, target and direct complication-definition variables;
- fits MAT-Appendix exactly once in each of five outer folds;
- creates genuine out-of-fold predictions;
- performs cross-fitted Platt calibration and cross-fitted threshold selection;
- calculates ECE, MCE, Brier score, log loss and 1,000-replicate bootstrap confidence intervals;
- creates ROC, PR, calibration, confusion-matrix, missingness and XAI figures;
- produces fold-aggregated permutation XAI plus representative-fold SHAP and LIME;
- writes a compressed reproducibility PKL containing the five CPU model state dictionaries;
- saves tables, 300-DPI PNG/PDF figures, JSON results, a model card, checksums and a paper-assets ZIP;
- optionally commits the completed run under `results/runs/<RUN_ID>/` and updates `results/LATEST_RUN.txt`.

## GitHub auto-save from Colab

Create a fine-grained GitHub token restricted to `AzizulHakim00/MAT-Appendix` with **Contents: Read and write** permission. In Colab, open **Secrets**, add a secret named `GITHUB_TOKEN`, and enable notebook access.

The token is read through Colab Secrets, is never printed, and is not written to the PKL, figures, logs or repository files.

## Reusing results without retraining

After the primary run has been pushed, open `MAT_Appendix_Load_Artifacts.ipynb`. It reads `results/LATEST_RUN.txt`, downloads the corresponding PKL and exposes:

- the exact five fold indices for fair future baselines;
- fold-specific preprocessing dictionaries;
- five MAT-Appendix state dictionaries;
- raw and calibrated OOF predictions;
- calibration objects and thresholds;
- metrics, confidence intervals and XAI arrays.

Future baseline, ablation and robustness notebooks should reuse the saved folds and compare against the saved MAT-Appendix OOF predictions. The proposed model does not need to be trained again.

## Source layout

- `src/part_01.pyfrag` — configuration, direct dataset fetch, leakage audit, preprocessing and model definition.
- `src/part_02.pyfrag` — five-fold training, OOF calibration, confidence intervals and XAI computation.
- `src/part_03.pyfrag` — figures, tables, PKL/ZIP generation and secure GitHub result push.
- `docs/REPRODUCIBILITY.md` — fold, leakage and artifact reuse contract.

The three source fragments concatenate into one syntax-checked Python program. Their split only keeps the GitHub/Colab bootstrap lightweight.

## Dataset

Regensburg Pediatric Appendicitis, UCI dataset ID 938 / Zenodo DOI `10.5281/zenodo.7711412`.

## Scientific scope

This is retrospective research code, not a clinical decision system. The primary endpoint is severity among records labeled complicated or uncomplicated appendicitis. The notebook intentionally excludes direct target-defining and post-decision features.
