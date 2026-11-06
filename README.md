# StreamGraph

**A Scalable Spark-Based Framework for Domain-Invariant Graph Learning Across Non-Stationary Data Streams**

Working application: financial graph analysis, beginning with illicit-transaction classification on the original Elliptic Bitcoin dataset.

## Current milestone

Milestone 1 implements data acquisition, integrity checks, chronological splits, random-forest/MLP baselines, validation reporting, and graph-snapshot export. It does not yet implement GraphSAGE, DANN/GRL, or Spark processing. Those components will build on the same validated data and protocol.

The `artifacts/` directory contains an actual initial run on Elliptic, not synthetic example scores. Open the run's `report.html` to see it. This run uses one seed and reports **validation/model-selection measurements**. The final test period has not been scored.

## Quick start on macOS with VS Code

1. Clone this repository or extract the starter archive, then open the `streamgraph` folder in VS Code.
2. Select **Terminal > New Terminal**. Check that the terminal is inside this folder.
3. Run:

```bash
bash scripts/run_macos.sh
```

The script creates `.venv`, installs the baseline dependencies, runs integrity tests, downloads the three CSVs from the PyG-maintained mirror, and runs the experiment. The raw dataset and numerical cache require approximately 1 GB of working disk space; models and outputs need additional space. The first download includes a roughly 151 MB feature archive.

The project requires Python 3.11 or newer. Your existing Python 3.13 installation can be used for this baseline stage. Saved runs record their Python and package versions in `results.json`. Java and a GPU are not needed for this milestone; Java is needed for the later Spark stage.

After the script finishes, select the `.venv` interpreter in VS Code and open `artifacts/baseline_*/report.html` in a browser. Each report is a standalone HTML file with embedded charts.

## Manual setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python -m unittest discover -s tests -v
streamgraph run
```

For later terminal sessions:

```bash
source .venv/bin/activate
streamgraph doctor
```

## Commands

### macOS HTTPS certificate error

If a python.org macOS installation reports `CERTIFICATE_VERIFY_FAILED: unable to get local issuer certificate`, complete its certificate setup. For Python 3.13 run:

```bash
/bin/sh "/Applications/Python 3.13/Install Certificates.command"
```

After `-- update complete`, activate the project environment and run `streamgraph run` again. This is the certificate setup described in the [official Python macOS guide](https://docs.python.org/3.13/using/mac.html). The Matplotlib first-run font-cache message is normal.

Version 0.1.1 adds certifi's certificate bundle to the verified download context and reports readable connection errors through the installed `streamgraph` command. Certificate and hostname verification remain enabled. After pulling this update into an existing environment, install the new dependency before retrying:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
streamgraph run
```

Run from the project directory, or use `streamgraph --root /absolute/project/path COMMAND`.

| Command | Behavior |
| --- | --- |
| `streamgraph doctor` | Show Python and installed baseline package versions. |
| `streamgraph download` | Download/resume the three official-mirror CSVs and record hashes. |
| `streamgraph prepare` | Validate the CSVs, cache graph arrays, and write a data audit. |
| `streamgraph audit` | Rebuild dataset and split statistics. |
| `streamgraph train` | Fit both source-only baselines; select epochs/thresholds on validation. |
| `streamgraph snapshot --step 1` | Export the original time-step graph as an NPZ file for the next GNN stage. |
| `streamgraph report` | Rebuild the latest completed run's standalone HTML report. |
| `streamgraph run` | Run the complete milestone-one workflow. |

`streamgraph download --force` replaces raw CSVs with fresh mirror downloads. Run `prepare` afterward to rebuild the cache.

## Protocol

The configured time ranges are proposed project settings, not an asserted reproduction of the dataset paper's split.

| Original time steps | Role | How labels are used now |
| --- | --- | --- |
| 1-16 | Source | Known labels train RF/MLP. |
| 17-24 | Historical adaptation prefix | Reserved for later DANN work; no task labels used now. |
| 25-34 | Validation | MLP checkpoint and prediction-threshold selection. |
| 35-49 | Final test | Held out from fitting, threshold selection, and model scoring in this milestone. |

The label mapping is `1 -> illicit (1)`, `2 -> licit (0)`, `unknown -> -1`. Unknown nodes stay available for permitted graph context, but never become negative training examples.

Transaction IDs are join keys, not predictors. Time step is used for chronology and later domain membership, not as an ordinary feature. The `all` feature view has 165 inputs after excluding time. The optional `local` view uses the first 93 remaining local fields. Copy the config to a new file and use `streamgraph --config configs/your_copy.json train` to preserve experiment provenance.

Scaling is fitted to known source training rows only. The MLP uses explicit epochs with validation AP stopping, instead of scikit-learn's random internal early-stopping split. Class weights are derived from source labels. The RF uses the same features and source labels.

Primary ranking metric: illicit **average precision (AP)**. Precision, recall, and F1 use a validation-selected threshold. These are development measurements, not final test results. A GNN will not necessarily outperform the random forest.

## Graph meaning and replay boundary

Nodes are transactions; directed edges are observed payment flows. The release has disconnected time-step graphs. `snapshot` preserves these edges and reindexes them locally; it never adds fabricated temporal edges.

The published data includes coarse time steps and neighborhood-aggregated features. Our later streaming demonstration will initially replay **completed historical snapshots**. This milestone does not establish that every feature was available for an instantaneous live financial decision.

## Project structure

```text
configs/baseline.json             explicit splits and model settings
src/streamgraph/config.py         chronological role validation
src/streamgraph/data.py           download, joins, audit, graph export
src/streamgraph/experiments.py    RF/MLP fitting and validation metrics
src/streamgraph/report.py         measured HTML charts/report
src/streamgraph/cli.py            reproducible command workflow
tests/                           research-integrity checks
notebooks/01_explore.ipynb        optional VS Code exploration
artifacts/                       measured initial run and audit
docs/                            next milestone and research protocol
```

Raw CSVs, the processed full dataset, snapshot arrays, and fitted models are excluded from Git. `streamgraph run` downloads the raw data and regenerates the cache, models, and reports locally. Recorded validation summaries are available under `artifacts/`.

## Next milestone

Implement GraphSAGE on the same feature view, source labels, and validation interval. Compare it with the MLP before adding DANN. Keep final-test scoring for a later protocol freeze. See `docs/NEXT_MILESTONE.md`.

## Primary sources

- [Elliptic official release](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set)
- [Dataset paper, Weber et al.](https://arxiv.org/abs/1908.02591)
- [Official PyG Elliptic loader and mirror reference](https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/datasets/elliptic.py)
- [GraphSAGE, Hamilton et al.](https://arxiv.org/abs/1706.02216)
- [Domain adaptation and gradient reversal, Ganin and Lempitsky](https://proceedings.mlr.press/v37/ganin15.html)

Dataset labels describe the published benchmark. They are not a claim that this prototype determines the legality of arbitrary transactions.
