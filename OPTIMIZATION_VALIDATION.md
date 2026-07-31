# MAT-Appendix Nested Optimization Validation

- Full optimization source: Python syntax compilation passed.
- Full-source notebook: valid nbformat JSON.
- Bootstrap notebook: valid nbformat JSON.
- Artifact-loader notebook: valid nbformat JSON.
- Encoded source reconstructs exactly to the validated 59,217-byte source.
- Decoded source SHA-256: `d0cba186ae89a29452d8b80c780f5f7cb7d3218d672fe1b203c24f087593b12d`.
- GitHub bootstrap verifies this checksum before execution.
- The pipeline uses five untouched outer folds and three-fold inner model/threshold/meta-model selection.
- Actual UCI training was not executed in the build environment; real metrics require the Colab run.
