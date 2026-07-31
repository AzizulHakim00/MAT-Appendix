# MAT-Appendix

**Missingness-Aware Feature-Token Transformer for leakage-safe pediatric appendicitis severity prediction.**

## Conference-complete notebook

[![Open Conference-Complete Notebook in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AzizulHakim00/MAT-Appendix/blob/main/MAT_Appendix_Conference_Complete.ipynb)

[![Load Latest Conference Artifacts](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AzizulHakim00/MAT-Appendix/blob/main/MAT_Appendix_Load_Conference_Artifacts.ipynb)

[![Upload Existing Drive Run](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AzizulHakim00/MAT-Appendix/blob/main/MAT_Appendix_Upload_Existing_Conference_Run.ipynb)

Run `MAT_Appendix_Conference_Complete.ipynb` for the paper analysis. It uses the clinically correct cohort:

- `Diagnosis == appendicitis`;
- target: complicated versus uncomplicated appendicitis;
- strict removal of identifiers, post-decision variables, procedures/pathology and direct complication-defining features;
- five identical stratified outer folds for every model;
- MAT-Appendix trained exactly once per outer fold.

## Model comparison

The conference notebook evaluates:

1. Logistic Regression
2. Random Forest
3. XGBoost
4. CatBoost
5. HistGradientBoosting
6. Ordinary FT-Transformer
7. MAT without learnable missingness embedding
8. Proposed MAT-Appendix

## Complete evaluation

Generated outputs include:

- accuracy and error rate;
- balanced accuracy;
- precision/PPV, recall/sensitivity/TPR, specificity/TNR and NPV;
- F1, F2, MCC and Cohen's kappa;
- ROC-AUC and PR-AUC;
- Brier score, log loss, ECE and MCE;
- FPR, FNR and TP/TN/FP/FN;
- 1,000-replicate bootstrap confidence intervals;
- cross-fitted calibration and cross-fitted operating thresholds;
- 90% and 95% sensitivity operating points;
- model-specific confusion matrices;
- paired bootstrap comparisons against every baseline;
- fold stability, ROC, PR, core-metric, calibration-error, probability-loss and error-rate figures;
- fold-aggregated permutation importance;
- SHAP value-versus-missingness aggregation;
- SHAP beeswarm and LIME local explanation;
- age, sex and patient-level missingness subgroup analysis.

## Reproducibility

The run writes `mat_appendix_complete_reproducibility.pkl`, containing exact folds, fold-specific preprocessing, five proposed-model CPU state dictionaries, all raw/calibrated OOF predictions, thresholds, calibration objects, full metrics, confidence intervals, paired-bootstrap results, XAI arrays, subgroup results, dataset SHA-256 and software versions.

After a successful GitHub export, `MAT_Appendix_Load_Conference_Artifacts.ipynb` loads the latest PKL without retraining.

## GitHub auto-save from Colab

Create a fine-grained token restricted to this repository with **Contents: Read and write**. In Colab Secrets, add `GITHUB_TOKEN` and enable notebook access. The token is never printed or written to artifacts.

The completed run is committed under:

```text
results/runs/<RUN_ID>/
results/LATEST_RUN.txt
```

If the run saves to Drive but GitHub export is skipped, use `MAT_Appendix_Upload_Existing_Conference_Run.ipynb`; it uploads the completed run without retraining.

## Source layout

The single Colab cell concatenates these syntax-checked fragments:

- `src/v2/part_01.pyfrag` through `src/v2/part_06.pyfrag`
- `src/v2/part_07_pre.pyfrag`
- `src/v2/part_07b.pyfrag`
- `src/v2/part_07_post.pyfrag`
- `src/v2/part_08.pyfrag`

## Earlier exploratory notebook

`MAT_Appendix_Primary_5Fold.ipynb` is retained only as an exploratory run. Its 781-record endpoint included uncomplicated/no-appendicitis negatives and must not be used as the primary severity result in the paper.

## Dataset and scientific scope

Regensburg Pediatric Appendicitis, UCI dataset ID 938 / Zenodo DOI `10.5281/zenodo.7711412`.

This is retrospective, single-center research software. It is not a clinical decision system, and results must not be described as clinical-deployment-ready or state-of-the-art without external validation and direct literature-matched comparisons.
