"""GraphSAGE reference on completed Elliptic snapshots, with no test scoring.

One graph is active at a time. This is a bounded model-training reference;
the NumPy dataset remains resident in host memory. It is not distributed GNN
training or a claim of instantaneous transaction-time inference.
"""

from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import resource
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import SAGEConv

from streamgraph.config import config_hash, split_masks
from streamgraph.data import Dataset, dataset_fingerprint, utc_now, write_json
from streamgraph.experiments import (binary_metrics, fit_source_scaler,
                                     require_both_classes, select_threshold)


def validate_graph_config(config: dict) -> None:
    for name in ["hidden_channels", "epochs", "patience", "cpu_threads"]:
        if type(config.get(name)) is not int or config[name] < 1:
            raise ValueError(f"graphsage.{name} must be a positive integer")
    if not 0 <= config.get("dropout", -1) < 1:
        raise ValueError("graphsage.dropout must be in [0, 1)")
    for name in ["learning_rate", "gradient_clip_norm"]:
        if config.get(name, 0) <= 0:
            raise ValueError(f"graphsage.{name} must be positive")
    if config.get("weight_decay", -1) < 0:
        raise ValueError("graphsage.weight_decay cannot be negative")
    if config.get("edge_mode") not in {"original", "none"}:
        raise ValueError("graphsage.edge_mode must be original or none")


class GraphSAGEClassifier(nn.Module):
    """Two incoming-neighbor mean aggregations plus a binary task head."""

    def __init__(self, in_channels: int, hidden_channels: int, dropout: float):
        super().__init__()
        self.convs = nn.ModuleList([SAGEConv(in_channels, hidden_channels, aggr="mean"),
                                    SAGEConv(hidden_channels, hidden_channels, aggr="mean")])
        self.classifier = nn.Linear(hidden_channels, 2)
        self.dropout = dropout

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = F.dropout(F.relu(conv(x, edge_index)), p=self.dropout, training=self.training)
        return x

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(x, edge_index))


def masked_task_loss(logits: torch.Tensor, labels: torch.Tensor,
                     class_weights: torch.Tensor) -> torch.Tensor:
    known = labels >= 0
    if not bool(known.any()):
        raise ValueError("A supervised snapshot needs at least one known label")
    if bool((labels[known] > 1).any()):
        raise ValueError("Known graph labels must be licit=0 or illicit=1")
    return F.cross_entropy(logits[known], labels[known], weight=class_weights)


@dataclass
class SnapshotLayout:
    time_step: int
    nodes: np.ndarray
    edges: np.ndarray


def snapshot_layout(dataset: Dataset, step: int, edge_mode: str = "original") -> SnapshotLayout:
    if not 1 <= step <= 49:
        raise ValueError("Snapshot time_step must be within 1-49")
    if edge_mode not in {"original", "none"}:
        raise ValueError("edge_mode must be original or none")
    nodes = np.flatnonzero(dataset.time_steps == step)
    if not len(nodes):
        raise ValueError(f"No nodes in snapshot {step}")
    if edge_mode == "none":
        return SnapshotLayout(step, nodes, np.empty((2, 0), dtype=np.int64))
    source, target = dataset.edge_index
    # Reject both incoming and outgoing cross-snapshot edges, not just the
    # subset whose sources happen to belong to this snapshot.
    touches = (dataset.time_steps[source] == step) | (dataset.time_steps[target] == step)
    selected_edges = dataset.edge_index[:, touches]
    mapping = np.full(len(dataset.labels), -1, dtype=np.int64)
    mapping[nodes] = np.arange(len(nodes))
    edges = mapping[selected_edges]
    if edges.size and bool((edges < 0).any()):
        raise ValueError("Cross-snapshot edges would import context from another period")
    return SnapshotLayout(step, nodes, edges)


def graph_tensors(dataset: Dataset, layout: SnapshotLayout, scaler, feature_view: str):
    x = scaler.transform(dataset.feature_view(feature_view)[layout.nodes]).astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError(f"Non-finite features in active snapshot {layout.time_step}")
    return (torch.from_numpy(x),
            torch.from_numpy(layout.edges.copy()).long(),
            torch.from_numpy(dataset.labels[layout.nodes].astype(np.int64)))


def score_snapshots(model, dataset, layouts, scaler, feature_view) -> pd.DataFrame:
    model.eval()
    records = []
    with torch.inference_mode():
        for layout in layouts:
            x, edge_index, y = graph_tensors(dataset, layout, scaler, feature_view)
            known = y >= 0
            score = model(x, edge_index).softmax(dim=-1)[known, 1].numpy()
            mask = known.numpy()
            records.append(pd.DataFrame({"transaction_id": dataset.transaction_ids[layout.nodes][mask],
                                          "time_step": layout.time_step,
                                          "true_illicit": y[known].numpy(),
                                          "graphsage_score": score}))
    if not records:
        raise ValueError("No validation snapshots")
    result = pd.concat(records, ignore_index=True)
    require_both_classes(result.true_illicit.to_numpy(), "Graph validation")
    return result


def matching_baselines(artifact_root: Path, protocol: dict, fingerprint: str) -> list:
    """Link only results with the same data, seed, feature view and protocol."""
    matches = []
    for path in sorted(artifact_root.glob("baseline_*/results.json")):
        result = json.loads(path.read_text())
        if (result.get("config_sha256") == config_hash(protocol)
                and result.get("dataset_fingerprint") == fingerprint
                and result.get("feature_view") == protocol["feature_view"]
                and result.get("seed") == protocol["seed"]
                and result.get("final_test_evaluated") is False):
            matches.append({"run": path.parent.name, "environment": result.get("environment"),
                            "validation_models": result["models"]})
    return matches


def peak_rss_mib() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024 ** 2 if platform.system() == "Darwin" else 1024)


def train_graphsage(dataset: Dataset, protocol: dict, settings: dict, artifact_root: Path) -> Path:
    validate_graph_config(settings)
    masks = split_masks(dataset.time_steps, dataset.labels, protocol)
    source_y = dataset.labels[masks["train"]]
    require_both_classes(source_y, "Graph source training")
    require_both_classes(dataset.labels[masks["validation_known"]], "Graph validation")
    torch.set_num_threads(settings["cpu_threads"])
    torch.manual_seed(protocol["seed"])
    torch.use_deterministic_algorithms(True)
    scaler = fit_source_scaler(dataset.feature_view(protocol["feature_view"]), masks["train"])
    def layouts_for(role):
        start, end = protocol["splits"][role]
        return [snapshot_layout(dataset, step, settings["edge_mode"])
                for step in range(start, end + 1) if bool((dataset.time_steps == step).any())]
    source_layouts, validation_layouts = layouts_for("source"), layouts_for("validation")
    combined = {"protocol": protocol, "graphsage": settings}
    fingerprint = dataset_fingerprint(dataset.raw_hashes)
    combined_hash = config_hash(combined)
    run_dir = artifact_root / (f"graphsage_seed{protocol['seed']}_{protocol['feature_view']}_"
                                f"{settings['edge_mode']}_{combined_hash[:8]}_{fingerprint[:8]}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "models").mkdir(exist_ok=True)
    write_json(run_dir / "config.json", combined)
    write_json(run_dir / "run_status.json", {"status": "running", "started_at": utc_now()})
    features = dataset.feature_view(protocol["feature_view"]).shape[1]
    model = GraphSAGEClassifier(features, settings["hidden_channels"], settings["dropout"])
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    counts = np.bincount(source_y, minlength=2)
    class_weights = torch.tensor(len(source_y) / (2 * counts), dtype=torch.float32)
    best_ap, best_epoch, best_state, waiting = -1.0, 0, None, 0
    history = []
    started = perf_counter()
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        losses = []
        for layout in source_layouts:
            x, edge_index, y = graph_tensors(dataset, layout, scaler, protocol["feature_view"])
            if not bool((y >= 0).any()):
                continue
            optimizer.zero_grad(set_to_none=True)
            loss = masked_task_loss(model(x, edge_index), y, class_weights)
            if not bool(torch.isfinite(loss)):
                raise ValueError("Non-finite graph training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip_norm"])
            optimizer.step()
            losses.append(float(loss.detach()))
        predictions = score_snapshots(model, dataset, validation_layouts, scaler, protocol["feature_view"])
        ap = float(average_precision_score(predictions.true_illicit, predictions.graphsage_score))
        history.append({"epoch": epoch, "mean_snapshot_training_loss": float(np.mean(losses)), "validation_average_precision": ap})
        print(f"GraphSAGE {settings['edge_mode']} epoch {epoch:02}: validation AP={ap:.4f}", flush=True)
        if ap > best_ap + 1e-6:
            best_ap, best_epoch, waiting = ap, epoch, 0
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        else:
            waiting += 1
        if waiting >= settings["patience"]:
            break
    fit_seconds = perf_counter() - started
    if best_state is None:
        raise RuntimeError("GraphSAGE did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    started = perf_counter()
    predictions = score_snapshots(model, dataset, validation_layouts, scaler, protocol["feature_view"])
    inference_seconds = perf_counter() - started
    threshold = select_threshold(predictions.true_illicit.to_numpy(), predictions.graphsage_score.to_numpy())
    metrics = binary_metrics(predictions.true_illicit.to_numpy(), predictions.graphsage_score.to_numpy(), threshold)
    predictions["graphsage_predicted_illicit"] = (predictions.graphsage_score >= threshold).astype(np.int8)
    step_scores = []
    for step, group in predictions.groupby("time_step"):
        step_scores.append({"time_step": int(step), "known_nodes": len(group),
                            "illicit_nodes": int(group.true_illicit.sum()),
                            "average_precision": float(average_precision_score(group.true_illicit, group.graphsage_score))
                            if group.true_illicit.nunique() == 2 else None})
    active = source_layouts + validation_layouts
    result = {"stage": "milestone_2_graphsage", "scope": "validation / model selection", "created_at": utc_now(),
              "seed": protocol["seed"], "feature_view": protocol["feature_view"], "edge_mode": settings["edge_mode"],
              "features": features, "config_sha256": combined_hash, "protocol_config_sha256": config_hash(protocol),
              "dataset_fingerprint": fingerprint, "final_test_evaluated": False,
              "fit_audit": {"source_rows": int(masks["train"].sum()),
                            "source_snapshot_steps": [x.time_step for x in source_layouts],
                            "validation_snapshot_steps": [x.time_step for x in validation_layouts],
                            "scaler_fit_max_time_step": int(dataset.time_steps[masks["train"]].max()),
                            "unknown_labels_used_in_supervised_fit": 0,
                            "adaptation_prefix_used": False, "final_test_rows_used_in_fit_or_prediction": 0,
                            "training_transaction_ids_sha256": hashlib.sha256(dataset.transaction_ids[masks["train"]].tobytes()).hexdigest()},
              "validation": metrics, "per_time_step": step_scores,
              "training": {"best_epoch": best_epoch, "epochs_run": len(history),
                           "fit_with_validation_seconds": fit_seconds, "validation_graph_inference_seconds": inference_seconds,
                           "trainable_parameters": sum(p.numel() for p in model.parameters())},
              "resources": {"peak_process_rss_mib": peak_rss_mib(), "device": "cpu",
                            "max_active_snapshot_nodes": max(len(x.nodes) for x in active),
                            "max_active_snapshot_edges": max(x.edges.shape[1] for x in active),
                            "full_numpy_dataset_resident_in_host_memory": True},
              "environment": {"python": platform.python_version(), "platform": platform.platform(),
                              **{x: importlib.metadata.version(x) for x in ["numpy", "pandas", "scikit-learn", "torch", "torch-geometric"]}},
              "matching_tabular_baselines": matching_baselines(artifact_root, protocol, fingerprint),
              "notes": ["One initial seed; validation chooses checkpoint and threshold.",
                        "Original edges are directed payment flows; no cross-step edges are invented.",
                        "Completed-snapshot benchmark; not a live as-of transaction decision.",
                        "This is a CPU reference, not distributed GNN training or a scalability result."]}
    predictions.to_csv(run_dir / "validation_predictions.csv", index=False)
    pd.DataFrame(history).to_csv(run_dir / "learning_curve.csv", index=False)
    write_json(run_dir / "results.json", result)
    # Tensor-only model/scaler data supports torch.load(..., weights_only=True).
    torch.save({"state_dict": best_state, "in_channels": features, "hidden_channels": settings["hidden_channels"],
                "dropout": settings["dropout"], "threshold": threshold,
                "scaler_mean": torch.from_numpy(scaler.mean_.copy()),
                "scaler_scale": torch.from_numpy(scaler.scale_.copy())}, run_dir / "models/graphsage.pt")
    write_json(run_dir / "run_status.json", {"status": "complete", "completed_at": utc_now(), "final_test_evaluated": False})
    print(f"GraphSAGE validation AP={metrics['average_precision']:.4f}; results: {run_dir}", flush=True)
    return run_dir
