# Generated results

No fabricated or unexecuted metric is stored here.

A successful primary Colab run creates `results/runs/<UTC_RUN_ID>/` and updates `LATEST_RUN.txt`. Each run may contain:

- `mat_appendix_reproducibility.pkl`;
- five fold state dictionaries;
- OOF predictions and metric tables;
- calibration bins and bootstrap confidence intervals;
- leakage and missingness audits;
- permutation, SHAP and LIME artifacts;
- 300-DPI PNG/PDF paper figures;
- JSON summaries, model card, software versions and checksums;
- a zipped paper-assets package.

Use `MAT_Appendix_Load_Artifacts.ipynb` after the first successful GitHub-pushed run.
