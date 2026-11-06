# Start here

1. Open the streamgraph folder in VS Code.
2. Open Terminal > New Terminal.
3. Run: bash scripts/run_macos.sh
4. Open artifacts/baseline_*/report.html to inspect the measured validation run.

The archive includes source code, 10 passing integrity tests, the real-data audit, validation predictions, initial fitted models, a standalone HTML results report, and a VS Code exploration notebook.

Raw data is downloaded automatically by the setup workflow. It is not bundled. The first run requires Python 3.11+ and about 1 GB of working disk space.

Implemented: data preparation and RF/MLP baselines. Next: GraphSAGE, then DANN/GRL, then Spark replay. Final test steps 35-49 have not been scored.
