# Conference-Complete Validation Report

## Completed checks

- Full Python source parsed and compiled with `py_compile`.
- Reconstructed fragment sequence compiled successfully.
- GitHub bootstrap notebook JSON validated.
- Full-source one-cell notebook JSON and code validated.
- Artifact loader notebook JSON and code validated.
- No-retraining GitHub uploader notebook JSON and code validated.
- Corrected cohort logic explicitly requires `Diagnosis == appendicitis` and a valid complicated/uncomplicated severity label.
- Leakage audit removes identifiers, target/post-decision variables, procedure/pathology variables and direct complication-defining variables.
- Every model uses the same outer folds.
- Proposed MAT-Appendix is fit exactly once in each of five outer folds.
- Full metric, confusion-matrix, calibration-error, error-rate, probability-loss, XAI, paired-bootstrap and subgroup outputs are included.

## Not completed in this environment

The real corrected-cohort experiment was not executed here because this runtime has no external dataset access or Colab GPU. Real performance claims must come from the executed conference-complete Colab notebook, not from the earlier 781-record exploratory result.

## Required scientific wording

The generated `LIMITATIONS.md` states that the study is retrospective, single-center, lacks external validation, may encode local missingness workflows, and must not be described as clinical-deployment-ready or state-of-the-art.
