# Milestone 2: initial validation measurements

All rows use source steps 1-16, validation steps 25-34, seed 42, and 165 numeric features. Validation selected checkpoints and thresholds. Final test steps 35-49 were not scored.

| Model | Validation AP | Illicit precision | Illicit recall | Illicit F1 |
| --- | ---: | ---: | ---: | ---: |
| Random forest | 0.7512 | 0.9075 | 0.6396 | 0.7504 |
| MLP | 0.5662 | 0.6759 | 0.5704 | 0.6187 |
| GraphSAGE, original directed edges | 0.4939 | 0.5668 | 0.6010 | 0.5834 |
| GraphSAGE, edge-free control | 0.5475 | 0.6520 | 0.5728 | 0.6099 |

The initial GraphSAGE configuration is below the random forest. The edge-free control also exceeds this original-edge configuration. These measurements establish a working graph-learning reference, not a predictive improvement. Further tuning and domain-adaptation experiments must retain matched information budgets and the final holdout.

Graph runs fit scaling to 16,044 known source labels and evaluate 6,288 known validation labels. Unknown nodes remain graph context without supervised targets. Dataset/protocol fingerprints and source training-ID hashes match the recorded tabular baselines.

The original-edge run selected epoch 8 of 14, with 29,570 parameters. Its maximum active snapshot has 7,880 nodes and 9,164 edges. Host RAM also contains the full NumPy dataset; this is not a distributed-scaling measurement.

Graph environment: Python 3.12.14, PyTorch 2.12.0+cpu, PyG 2.8.1, CPU. Saved tabular runs record their own environments. This is an initial one-seed development comparison, not a statistical conclusion.

Reproduce with `streamgraph train-graph` and `streamgraph train-graph --edges none`. Compact configs, curves and result JSONs are under `artifacts/graphsage_*`. Binary models and per-node predictions are generated locally.
