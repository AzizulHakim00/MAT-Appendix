# Reproducibility contract

The proposed MAT-Appendix model is trained exactly five times: one model in each outer stratified fold. No proposed-model retraining is performed for calibration, confidence intervals or XAI.

The generated `mat_appendix_reproducibility.pkl` stores:

- dataset ID, filtered row IDs, feature names and dataset SHA-256 fingerprint;
- the exact five train/test index sets;
- fold-specific preprocessing dictionaries fitted only on each outer-training fold;
- five CPU PyTorch state dictionaries and their OOF predictions;
- cross-fitted Platt calibrators, per-sample cross-fitted thresholds and a final deployment calibrator;
- fold and aggregate metrics, 1,000-bootstrap confidence intervals, ECE/MCE and Brier score;
- fold-aggregated permutation importance, representative-fold SHAP arrays and LIME explanation data;
- Python/package/device versions.

Future baseline, ablation and robustness experiments must reuse the saved outer folds and compare against the saved OOF predictions. The dataset fingerprint must match before comparison.

## Leakage policy

Before splitting, the primary analysis removes identifiers, target columns, post-decision variables, procedure/pathology fields and variables whose names directly encode the complicated-appendicitis definition, including perforation, abscess, gangrene, complication and peritonitis indicators. Constant/empty columns are also removed. All learned preprocessing parameters remain inside each outer-training fold.
