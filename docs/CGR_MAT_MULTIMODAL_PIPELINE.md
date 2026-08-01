# CGR-MAT v1 Multimodal Pipeline

## Purpose

This pipeline evaluates whether raw multiview ultrasound images add reproducible severity-prediction value beyond a leakage-safe tabular expert for confirmed pediatric appendicitis.

The v1 core is a **Clinical-Guided Residual Multiview Attention Transformer**. It deliberately does not add a generative or reinforcement-learning module before the image signal and residual-fusion contribution have passed ablation gates.

## Architecture

1. A CatBoost base expert uses only pre-decision, leakage-safe structured variables.
2. A pretrained ResNet-18 encodes up to six ultrasound views per patient.
3. A trainable quality-attention module pools variable-length image sets.
4. Tabular, image, base-logit and missing-modality tokens are fused by a transformer.
5. A learned gate applies a residual correction to the base logit.
6. Modality dropout trains the model to remain usable when imaging or structured information is missing.

The final logit is:

`final_logit = base_logit + gate × learned_residual`

## Scientific safeguards

- The official patient-level test codes define an untouched final holdout.
- The remaining development cohort uses one prespecified stratified five-fold outer CV.
- CatBoost training probabilities are cross-fitted inside each outer-training fold.
- Epoch selection uses a training-only monitor split, not the outer validation fold.
- Calibration and the secondary operating threshold use development OOF predictions only.
- The official holdout is evaluated once after the final model is fitted.
- `Peritonitis`, `Perforation`, `Appendicular_Abscess`, `Abscess_Location`, `Length_of_Stay`, outcome labels and identifiers are excluded from predictors.
- Cohort construction aborts unless the expected signature is N=463 with 118 complicated cases.
- Dataset files and source code are checksum verified.
- Padded ultrasound views are masked before the image backbone and cannot alter BatchNorm statistics.

## Run modes

- `smoke`: dependency, data, model and export test; not a scientific result.
- `quick`: shorter three-fold experiment for debugging.
- `full`: the prespecified five-fold scientific run.

The public Colab notebook defaults to `full` and requires a GPU runtime.

## Automatic data flow

The notebook downloads the official Zenodo record directly, verifies MD5 checksums, caches the archive in Google Drive, extracts images to local Colab storage, validates image readability, maps image filename prefixes to `US_Number`, constructs the confirmed-appendicitis severity cohort and applies the official holdout codes.

No manual upload and no Kaggle API token are required.

## Live notebook output

During execution the notebook displays:

- device and pinned-source verification;
- download and extraction progress;
- dataset and image audit table;
- development/holdout split table;
- live fold and epoch progress;
- fold metric tables;
- outer-CV mean, standard deviation, minimum and maximum;
- official holdout comparison for the CatBoost base and CGR-MAT;
- ROC and precision-recall curves;
- calibration plot;
- confusion matrix;
- tabular feature-importance figure.

## Saved artifacts

The deterministic source-aware run directory is:

`MyDrive/MAT-Appendix/cgr_mat_runs/cgr_mat_v1_<mode>_<hash>/`

Important outputs include:

- `config.json` and `environment_freeze.txt`
- `dataset_audit.json`, `cohort_manifest.csv`, `image_manifest.csv`, `split_summary.csv`
- fold-specific `model_best.pt`, `preprocessor.pkl`, `tabular_base_model.pkl`, `tabular_base_model.cbm`
- fold histories, predictions, metrics and figures
- `oof_predictions.csv`, `fold_metrics.csv`, `cv_summary_mean_std.csv`
- `temperature_scaler.pkl`, `temperature.json`, `operating_threshold.json`
- `final_cgr_mat_model.pt`
- `final_preprocessor.pkl`, `final_tabular_base_model.pkl`, `final_tabular_base_model.cbm`
- `official_holdout_predictions.csv`, `official_holdout_metrics.csv`, `official_holdout_metrics.json`
- ROC, PR, calibration, confusion-matrix and feature-importance PNG files
- `reproducibility_bundle.pkl`, `manifest_sha256.csv`
- a ZIP archive of the complete run directory.

Completed folds are resume-safe through `COMPLETED.json`. Setting `CGR_MAT_FORCE_RESTART=1` deliberately starts the same configuration again.

## Interpretation rule

The primary result is the untouched official holdout comparison. A higher training score or an epoch-selection monitor score is not evidence of generalisation. CGR-MAT should be retained as the proposed performance model only when it improves a prespecified endpoint such as holdout PR-AUC, Brier score, or sensitivity at a fixed specificity over the tabular base.
