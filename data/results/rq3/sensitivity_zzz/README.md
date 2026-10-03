# RQ3 results

This folder contains reproducible artifacts for the within-coda cross-clan transfer experiments. The primary score is paired transfer loss in bits per symbol: target-clan own score minus source-to-target score. Positive loss means the source predicts the target worse.

## Result hierarchy

- [calibration/](calibration/README.md): model selection and cross-fitted own-clan reference scores.
- [initial_caribbean/](initial_caribbean/README.md): EC1 ↔ EC2 initial comparison.
- [transfer_matrix/](transfer_matrix/README.md): full 9 × 9 matrix and its derived diagnostics.

## Headline findings

- All 72 cross-clan transfer losses are positive (range 0.317 to 2.815 bits/symbol).
- Coverage of an exact full timing pattern is low across cross-clan source-fold/target-coda evaluations; see the coverage summary.
- All 72 target-group bootstrap intervals lie above zero.
- Pacific non-identity codas generally transfer better than identity codas, but the exploratory Table S4 overlap association is opposite to the predeclared prediction.
