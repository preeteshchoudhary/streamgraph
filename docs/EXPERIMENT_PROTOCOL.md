# Recorded decisions

- Application: illicit-transaction classification as a financial graph research case study.
- Dataset: original Elliptic release via the PyG-maintained mirror.
- Label mapping: original 1 is positive; original 2 is negative; unknown is -1 and masked from supervised learning/metrics.
- ID joins: class rows are matched by transaction ID, never by row position.
- Features: exclude transaction ID and original time step. Default is all 165 numeric features. A 93-feature local view is a separately logged comparison.
- Splits: source 1-16; adaptation prefix 17-24; validation 25-34; test 35-49.
- Source labels only: baseline supervised training uses known labels in 1-16. Unlabelled/prefix/future labels are excluded.
- Validation: selects MLP checkpoint and operating threshold. Threshold-dependent validation scores are development measurements.
- Final test: no prediction or evaluation in milestone one.
- Preprocessing: StandardScaler fits known source training rows only; RF uses unscaled features.
- Imbalance: training-derived class/sample weights.
- Metrics: positive-class average precision, precision/recall/F1, balanced accuracy, and confusion matrix.
- Epoch stopping: manual chronological validation, not an internal random holdout.
- Initial run: one fixed seed, 42. Repeat across seeds after the model comparison is finalized.
- Graph: transaction nodes, original directed edges, no cross-step edges invented.
- Replay: later completed-snapshot demonstration; instantaneous transaction-time feature availability is not established.
- Provenance: raw file hashes, dataset/config fingerprints, training-ID hash, environment versions, and saved validation predictions.

## Reproducibility outputs

`artifacts/dataset_audit.json` gives the graph/label/time counts. Each `baseline_*` run contains its exact config, results, validation predictions, MLP learning curve, fitted models, and an embedded-chart HTML report. Raw files and processed full-dataset arrays are excluded from distribution and can be reproduced by `streamgraph run`.

## Required checks before later final-test use

Freeze feature view, hyperparameters, label access, threshold-selection rule, graph construction, and update policy using training/validation only. Keep a frozen-model test arm separate from a test-before-update adaptation arm. Log the latter's simulated label delay, model versions, and update budget. This is an experimental protocol decision, not a claim that an improvement is already established.
