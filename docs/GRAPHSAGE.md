# Milestone 2: the GraphSAGE reference

This milestone adds a two-layer PyTorch Geometric GraphSAGE encoder and binary classification head. It uses original directed payment-flow edges, incoming-neighbor mean aggregation, and node features. Training and evaluation operate on one completed original snapshot at a time.

## Run on a supported Mac or Linux environment

```bash
bash scripts/run_graph.sh
```

The script installs the pinned optional graph stack, runs integrity tests, prepares the dataset if needed, and executes `streamgraph train-graph`. Linux uses the official CPU PyTorch wheel index. The pinned PyTorch 2.12 macOS wheels require macOS 14+ and native Apple Silicon Python; the script checks this before installing. Intel Mac users should use a compatible Linux environment for this pinned milestone. The tabular baseline remains available separately.

For an existing supported environment:

```bash
source .venv/bin/activate
python -m pip install -r requirements-graph.txt
python -m pip install --no-deps -e .
streamgraph train-graph
```

The optional package extra is also available as `pip install -e '.[graph]'`; `requirements-graph.txt` additionally pins the baseline packages for reproducible experiments. No compiled neighbor-sampling extensions are required for this full-snapshot reference.

## Run through GitHub instead of a local graph installation

After this workflow is merged into the repository's default branch, open **Actions > GraphSAGE experiment > Run workflow**. Select `original` or `none` for the edge mode. The manual workflow runs on a Linux CPU runner, downloads the benchmark, executes integrity checks, and evaluates validation only. Download the generated result/model artifact from that run. Local PyTorch installation is not required for this browser-driven route; this is useful for Intel Macs and other machines outside the pinned local wheel support.

## Evaluation protocol

- Source labels and scaler fitting: known labels in steps 1-16 only.
- Adaptation prefix, steps 17-24: unused in this reference; reserved for later DANN work.
- Validation and checkpoint/threshold selection: steps 25-34.
- Final test, steps 35-49: no graph construction, feature transformation, model prediction, or evaluation in this experiment.
- Unknown labels: available as graph context, excluded from task loss and label-based metrics.
- Original time steps remain disconnected. No cross-period links or synthetic payment flows are created.
- Default feature view: the same 165 numeric fields as RF/MLP, excluding transaction ID and time step.

GraphSAGE's root transform already uses the node's own features. We do not insert extra self-loop edges. The default uses original directed edges and does not silently symmetrize them.

## Test whether edges help

```bash
streamgraph train-graph --edges none
```

This is an edge-free control using the same model family, source labels, features, optimizer settings, seed, and validation rules. It tests the value of message passing more directly than comparing models with different parameter counts. Graph relationships are not guaranteed to improve a particular benchmark.

## Outputs

Each `artifacts/graphsage_*` run contains its protocol/model config, validation metrics, per-time-step AP, training curve, known-label validation predictions, and a best-checkpoint model. Binary checkpoints and per-node predictions are excluded from Git; compact metrics and learning curves may be committed for review.

The checkpoint contains tensors and primitive settings and can be read with `torch.load(path, weights_only=True)`. Stored scaler statistics originate from source training rows only.

Matching RF/MLP summaries are linked only when dataset fingerprint, protocol config hash, feature view, and seed match. Execution environments are recorded and can differ between saved runs. One seed and validation-selected thresholds are development measurements, not a final test result or a statistical superiority claim.

## Resource boundary

The active graph is bounded by an original snapshot. The full NumPy dataset still resides in host RAM. Peak RSS describes the whole process, including data and Python libraries, rather than the model alone. This milestone does not demonstrate distributed GNN training or Spark scalability. Sampling, Spark replay, and DANN/GRL remain subsequent work.
