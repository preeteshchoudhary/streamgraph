"""Source-only baselines; validation selects settings, final test stays unused."""

import copy
import hashlib
import importlib.metadata
from pathlib import Path
import platform
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits

from streamgraph.config import config_hash, split_masks
from streamgraph.data import Dataset, dataset_fingerprint, utc_now, write_json


def require_both_classes(labels: np.ndarray, role: str) -> None:
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError(f"{role} needs known licit and illicit labels; check the declared split")


def fit_source_scaler(features: np.ndarray, source_train_mask: np.ndarray) -> StandardScaler:
    if not source_train_mask.any():
        raise ValueError("No source training rows")
    return StandardScaler().fit(features[source_train_mask])


def select_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Select illicit F1 on validation only; AP never depends on this threshold."""
    require_both_classes(labels, "Validation")
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    tied = np.flatnonzero(np.isclose(f1, f1.max(), rtol=1e-10, atol=1e-12))
    return float(thresholds[tied[-1]])


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    require_both_classes(labels, "Evaluation")
    predictions = (probabilities >= threshold).astype(np.int8)
    return {"average_precision": float(average_precision_score(labels, probabilities)),
            "precision_illicit": float(precision_score(labels, predictions, zero_division=0)),
            "recall_illicit": float(recall_score(labels, predictions, zero_division=0)),
            "f1_illicit": float(f1_score(labels, predictions, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
            "threshold": float(threshold),
            "confusion_matrix_labels_0_1": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
            "known_nodes": int(len(labels)), "illicit_nodes": int((labels == 1).sum())}


def positive_probabilities(model, features: np.ndarray) -> np.ndarray:
    positive_column = int(np.flatnonzero(model.classes_ == 1)[0])
    return model.predict_proba(features)[:, positive_column]


def train_baselines(dataset: Dataset, config: dict, artifact_root: Path) -> Path:
    masks = split_masks(dataset.time_steps, dataset.labels, config)
    train_mask, validation_mask = masks["train"], masks["validation_known"]
    train_y, validation_y = dataset.labels[train_mask], dataset.labels[validation_mask]
    require_both_classes(train_y, "Source training")
    require_both_classes(validation_y, "Validation")
    features = dataset.feature_view(config["feature_view"])
    seed = config["seed"]
    cfg_hash = config_hash(config)
    data_hash = dataset_fingerprint(dataset.raw_hashes)
    run_dir = artifact_root / f"baseline_seed{seed}_{config['feature_view']}_{cfg_hash[:8]}_{data_hash[:8]}"
    run_dir.mkdir(parents=True, exist_ok=True)
    model_dir = run_dir / "models"
    model_dir.mkdir(exist_ok=True)
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "run_status.json", {"status": "running", "started_at": utc_now()})
    fit_audit = {"source_rows": int(train_mask.sum()),
                 "min_source_step": int(dataset.time_steps[train_mask].min()),
                 "max_source_step": int(dataset.time_steps[train_mask].max()),
                 "unknown_labels_used_in_supervised_fit": 0,
                 "adaptation_labels_used_in_supervised_fit": 0,
                 "validation_labels_used_in_supervised_fit": 0,
                 "final_test_rows_used_in_fit_or_prediction": 0,
                 "training_transaction_ids_sha256": hashlib.sha256(
                     dataset.transaction_ids[train_mask].tobytes()).hexdigest()}
    predictions = pd.DataFrame({"transaction_id": dataset.transaction_ids[validation_mask],
                                "time_step": dataset.time_steps[validation_mask],
                                "true_illicit": validation_y})
    results = {"stage": "milestone_1", "scope": "validation / model selection",
               "created_at": utc_now(), "seed": seed, "feature_view": config["feature_view"],
               "features": features.shape[1], "config_sha256": cfg_hash,
               "dataset_fingerprint": data_hash, "fit_audit": fit_audit,
               "final_test_evaluated": False,
               "threshold_rule": "maximum validation illicit F1; highest threshold breaks ties",
               "notes": ["One initial seed; not a multi-seed research conclusion.",
                         "Validation selects epochs and thresholds; scores are not held-out test performance.",
                         "Baselines use only the source interval; the adaptation prefix is reserved for later DANN work."],
               "environment": {"python": platform.python_version(),
                               **{n: importlib.metadata.version(n) for n in ["numpy", "pandas", "scikit-learn"]}},
               "models": {}}

    print(f"Source training: {len(train_y):,} known nodes; validation: {len(validation_y):,}; features: {features.shape[1]}", flush=True)
    print("Training random forest...", flush=True)
    forest = RandomForestClassifier(**config["random_forest"], random_state=seed,
                                     class_weight="balanced_subsample")
    started = perf_counter()
    forest.fit(features[train_mask], train_y)
    fit_seconds = perf_counter() - started
    started = perf_counter()
    probability = positive_probabilities(forest, features[validation_mask])
    predict_seconds = perf_counter() - started
    threshold = select_threshold(validation_y, probability)
    metrics = binary_metrics(validation_y, probability, threshold)
    metrics.update({"fit_seconds": fit_seconds, "validation_prediction_seconds": predict_seconds})
    results["models"]["random_forest"] = metrics
    predictions["random_forest_score"] = probability
    predictions["random_forest_predicted_illicit"] = (probability >= threshold).astype(np.int8)
    joblib.dump({"model": forest, "scaler": None, "config": config,
                 "dataset_fingerprint": data_hash, "threshold": threshold}, model_dir / "random_forest.joblib", compress=3)
    print(f"Random forest validation AP: {metrics['average_precision']:.4f}", flush=True)

    print("Training MLP with source-fitted scaling and chronological validation stopping...", flush=True)
    scaler = fit_source_scaler(features, train_mask)
    train_x = scaler.transform(features[train_mask]).astype(np.float32)
    validation_x = scaler.transform(features[validation_mask]).astype(np.float32)
    settings = config["mlp"]
    classifier = MLPClassifier(hidden_layer_sizes=tuple(settings["hidden_layer_sizes"]),
                              batch_size=settings["batch_size"],
                              learning_rate_init=settings["learning_rate_init"],
                              alpha=settings["alpha"], random_state=seed,
                              max_iter=1, early_stopping=False, shuffle=True)
    weights = compute_sample_weight(class_weight="balanced", y=train_y)
    best_model, best_ap, best_epoch, waiting = None, -1.0, 0, 0
    history = []
    started = perf_counter()
    with threadpool_limits(limits=config["random_forest"]["n_jobs"]):
        for epoch in range(1, settings["epochs"] + 1):
            classifier.partial_fit(train_x, train_y, classes=np.array([0, 1]), sample_weight=weights)
            score = float(average_precision_score(validation_y, positive_probabilities(classifier, validation_x)))
            history.append({"epoch": epoch, "training_loss": float(classifier.loss_), "validation_average_precision": score})
            if score > best_ap + 1e-6:
                best_model, best_ap, best_epoch, waiting = copy.deepcopy(classifier), score, epoch, 0
            else:
                waiting += 1
            print(f"MLP epoch {epoch:02}: validation AP={score:.4f}", flush=True)
            if waiting >= settings["patience"]:
                break
    fit_seconds = perf_counter() - started
    if best_model is None:
        raise RuntimeError("MLP did not produce a valid checkpoint")
    started = perf_counter()
    with threadpool_limits(limits=config["random_forest"]["n_jobs"]):
        probability = positive_probabilities(best_model, validation_x)
    predict_seconds = perf_counter() - started
    threshold = select_threshold(validation_y, probability)
    metrics = binary_metrics(validation_y, probability, threshold)
    metrics.update({"fit_and_validation_selection_seconds": fit_seconds,
                    "validation_prediction_seconds": predict_seconds,
                    "best_epoch": best_epoch, "epochs_run": len(history),
                    "scaler_fit_max_time_step": fit_audit["max_source_step"]})
    results["models"]["mlp"] = metrics
    predictions["mlp_score"] = probability
    predictions["mlp_predicted_illicit"] = (probability >= threshold).astype(np.int8)
    pd.DataFrame(history).to_csv(run_dir / "mlp_learning_curve.csv", index=False)
    joblib.dump({"model": best_model, "scaler": scaler, "config": config,
                 "dataset_fingerprint": data_hash, "threshold": threshold}, model_dir / "mlp.joblib", compress=3)
    predictions.to_csv(run_dir / "validation_predictions.csv", index=False)
    write_json(run_dir / "results.json", results)
    write_json(run_dir / "run_status.json", {"status": "complete", "completed_at": utc_now(),
                                            "final_test_evaluated": False})
    print(f"Saved results to {run_dir}; final test steps were not scored.", flush=True)
    return run_dir
