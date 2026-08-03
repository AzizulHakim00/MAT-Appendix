# MAT-Appendix V3.2 Locked Revalidation

This branch performs the final, preregistered revalidation of the strongest earlier MAT-Appendix candidate. It does not add another architecture after observing results.

## Direct Colab

Open `MAT_Appendix_V32_Locked_Revalidation.ipynb` from this branch and choose **Runtime → Run all**.

## Locked protocol

- Corrected cohort: 463 confirmed appendicitis patients, 118 complicated and 345 uncomplicated.
- Strict removal of post-decision and direct complication-defining variables.
- Five repeats of five-fold stratified outer validation: 25 untouched outer evaluations.
- Three inner folds generate level-1 OOF predictions inside each outer-training partition.
- Same outer splits for all candidates.
- Candidates: calibrated CatBoost, calibrated GlobalBlend, and V32LockedBlend.
- V32LockedBlend ranks only outer-training OOF prediction streams, selects exactly five, cross-fits Platt calibration, and searches non-negative sum-to-one weights using a fixed probability objective.
- Balanced and 90%-sensitivity thresholds are selected only from outer-training OOF predictions.
- No level-1 fallback is allowed; any fallback stops the scientific run.
- Primary evidence: mean ± sample SD across five complete repeats.
- Secondary evidence: patient consensus and 3,000-replicate paired stratified bootstrap.

## Predeclared stop rule

The performance-paper path continues only when V32LockedBlend simultaneously reaches:

- ROC-AUC ≥ 0.89
- PR-AUC ≥ 0.72
- balanced accuracy ≥ 0.82
- F1 ≥ 0.72
- Brier ≤ 0.118

It must also show paired probability superiority over GlobalBlend without a material balanced-accuracy loss. Otherwise the code automatically recommends freezing architecture development and using the work as a rigorous benchmark/comparative study or pivoting to a new dataset/task.

## Saved outputs

Google Drive:

```text
MyDrive/MAT-Appendix/v32_locked_runs/v32_locked_<config-hash>/
```

Important files:

- `summary_mean_std.csv`
- `repeat_metrics.csv`
- `consensus_metrics.csv`
- `paired_bootstrap.csv`
- `final_decision.json`
- `publication_audit.json`
- `all_repeated_nested_predictions.csv`
- `v32_locked_revalidation_complete_results.pkl`

Latest PKL copy:

```text
MyDrive/MAT-Appendix/v32_locked_runs/LATEST_V32_LOCKED_REVALIDATION.pkl
```

Completed outer folds are checkpointed and reused after interruption.
