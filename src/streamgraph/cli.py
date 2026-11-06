"""Command line workflow. Run from the project folder or pass --root."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys

from streamgraph.config import load_config
from streamgraph.data import (audit, download, export_snapshot, load_dataset,
                              load_prepared, save_prepared, write_json)
from streamgraph.experiments import train_baselines
from streamgraph.report import build_report


def doctor() -> None:
    print("Python:", platform.python_version(), sys.executable)
    print("Platform:", platform.system(), platform.machine(), platform.mac_ver()[0])
    for name in ["numpy", "pandas", "scikit-learn", "matplotlib"]:
        print(name + ":", importlib.metadata.version(name))
    print("Java:", shutil.which("java") or "not installed (needed at the later Spark stage)")
    for name in ["torch", "torch-geometric"]:
        try:
            print(name + ":", importlib.metadata.version(name))
        except importlib.metadata.PackageNotFoundError:
            print(name + ": not installed (optional GraphSAGE dependency)")
    print("Baseline commands use CPU Python. GraphSAGE is optional; Spark integration is a later milestone.")


def run_workflow(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="streamgraph", description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project directory")
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.json"))
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Show the installed baseline environment")
    fetch = commands.add_parser("download", help="Download official-mirror Elliptic files")
    fetch.add_argument("--force", action="store_true")
    commands.add_parser("prepare", help="Validate raw data and cache numerical graph arrays")
    commands.add_parser("audit", help="Write and print the dataset and split audit")
    commands.add_parser("train", help="Fit RF/MLP; evaluate validation only")
    graph = commands.add_parser("train-graph", help="Fit CPU GraphSAGE; evaluate chronological validation only")
    graph.add_argument("--graph-config", type=Path, default=Path("configs/graphsage.json"))
    graph.add_argument("--edges", choices=["original", "none"], help="Override the graph-edge ablation setting")
    commands.add_parser("run", help="Download, prepare, audit, train, report, and export snapshot 1")
    snapshot = commands.add_parser("snapshot", help="Export one original graph snapshot as NPZ")
    snapshot.add_argument("--step", type=int, default=1)
    report = commands.add_parser("report", help="Rebuild a report from saved measurements")
    report.add_argument("--run-dir", type=Path)
    args = parser.parse_args(argv)
    if args.command == "doctor":
        doctor(); return
    root = args.root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = load_config(config_path)
    raw = root / "data/raw"
    prepared = root / "data/processed/elliptic.npz"
    audit_path = root / "artifacts/dataset_audit.json"
    if args.command in {"download", "run"}:
        download(raw, force=getattr(args, "force", False))
        if args.command == "download": return
    if args.command in {"prepare", "run"}:
        print("Loading raw CSVs and validating the graph...", flush=True)
        dataset = load_dataset(raw)
        print("Saving validated numerical graph cache...", flush=True)
        save_prepared(dataset, prepared)
    elif args.command != "report":
        dataset = load_prepared(prepared) if prepared.exists() else load_dataset(raw)
    if args.command in {"prepare", "audit", "run"}:
        summary = audit(dataset, config)
        write_json(audit_path, summary)
        print(json.dumps({k: v for k, v in summary.items() if k != "per_time_step"}, indent=2), flush=True)
        if args.command in {"prepare", "audit"}: return
    if args.command == "snapshot":
        output = root / f"data/processed/snapshot_{args.step:02}.npz"
        print(json.dumps(export_snapshot(dataset, args.step, output, config["feature_view"]), indent=2)); return
    if args.command == "train-graph":
        try:
            from streamgraph.graph import train_graphsage
        except ModuleNotFoundError as error:
            raise ValueError("Install the optional graph stack with scripts/run_graph.sh or requirements-graph.txt") from error
        graph_path = args.graph_config if args.graph_config.is_absolute() else root / args.graph_config
        settings = json.loads(graph_path.read_text())
        if args.edges:
            settings["edge_mode"] = args.edges
        train_graphsage(dataset, config, settings, root / "artifacts")
    elif args.command in {"train", "run"}:
        write_json(audit_path, audit(dataset, config))
        run_dir = train_baselines(dataset, config, root / "artifacts")
        print("Report:", build_report(audit_path, run_dir))
        print(json.dumps(export_snapshot(dataset, 1, root / "data/processed/snapshot_01.npz", config["feature_view"]), indent=2))
    elif args.command == "report":
        candidates = list((root / "artifacts").glob("baseline_*/results.json"))
        run_dir = args.run_dir
        if run_dir is None:
            if not candidates:
                raise FileNotFoundError("No completed run; run streamgraph train first")
            run_dir = max(candidates, key=lambda p: p.stat().st_mtime).parent
        elif not run_dir.is_absolute():
            run_dir = root / run_dir
        print("Report:", build_report(audit_path, run_dir))


def main(argv=None) -> None:
    try:
        run_workflow(argv)
    except (FileNotFoundError, ValueError, OSError) as error:
        print(f"StreamGraph: {error}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
