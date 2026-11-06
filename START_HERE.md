# Start here

1. Open the streamgraph folder in VS Code.
2. Open Terminal > New Terminal.
3. Run bash scripts/run_macos.sh for the tabular baseline workflow.
4. Open artifacts/baseline_*/report.html to inspect the measured validation run.

The baseline environment runs 13 integrity checks. Installing the optional graph stack adds 8 GraphSAGE checks, for 21 checks in total.

For the graph stage on a supported Mac/Linux environment, run bash scripts/run_graph.sh. See docs/GRAPHSAGE.md for platform requirements and the edge-free control.

Data and models are regenerated locally; allow about 1 GB of working disk space plus space for the optional PyTorch environment. Implemented: data preparation, RF/MLP, and a CPU GraphSAGE reference. Next: DANN/GRL and Spark replay. Final test steps 35-49 have not been scored.
