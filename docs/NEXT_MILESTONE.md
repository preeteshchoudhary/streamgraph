# Milestone 2: a matched GraphSAGE baseline

The goal is to establish whether observed graph relationships add predictive value beyond exactly the same node features.

1. Add a compatible CPU PyTorch/PyTorch Geometric environment. Verify installation and neighbor sampling before freezing versions.
2. Read the prepared arrays. Keep source steps 1-16, validation steps 25-34, and the current known-label masks unchanged.
3. Fit the scaler on the same source rows as the MLP. Apply it without refitting to the allowed validation snapshot.
4. Build a two-layer mean-aggregation GraphSAGE candidate, initially 64 hidden units, dropout, and a binary task head.
5. Begin with each complete source snapshot; these graphs are small enough to establish a correct full-snapshot reference before sampling.
6. Compute supervised loss only for known source labels. Unknown source nodes may pass messages, but carry no task targets.
7. Use observed directed payment edges and record the message direction. Any symmetrized-edge experiment must be a separately logged ablation.
8. Select model checkpoint and threshold on chronological validation only. Predict a completed validation snapshot using its permitted context; do not attach future snapshots to training.
9. Compare RF, MLP, and GraphSAGE on average precision, illicit recall/F1, training time, inference cost, and memory under matched inputs.
10. Add neighbor sampling and compare it with the full-snapshot reference. Only then add the domain classifier and GRL.

## DANN stage after that

Use labelled source snapshots and the designated unlabelled historical adaptation prefix (steps 17-24). Domain labels identify the historical blocks. Keep adaptation-prefix task labels hidden during fitting. The encoder receives a reversed domain gradient; the domain head receives the ordinary minimizing gradient.

The alternative source-only generalization experiment has a different task-label budget and must be evaluated as a separate arm. Do not quietly pool the adaptation prefix into supervised training.

## Spark stage after the offline comparison

Replay original snapshot records in order, validate/prepare them in PySpark, and hand completed graph batches to the GNN worker. Preserve model versions and idempotent prediction keys. Measure pipeline time separately from learning time. Spark does not automatically distribute the GNN.
