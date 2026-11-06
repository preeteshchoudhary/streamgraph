"""Download and validate the three official-mirror Elliptic CSV files.

Labels are joined by transaction ID; row ordering is never assumed. Unknown
labels use -1, so they cannot accidentally become the negative class.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import ssl
import tempfile
from urllib.error import URLError
from urllib.request import Request, urlopen
from zipfile import ZipFile

import certifi
import numpy as np
import pandas as pd

from streamgraph.config import ROLES, config_hash, split_masks


MIRROR = "https://data.pyg.org/datasets/elliptic"
FILES = (
    "elliptic_txs_features.csv",
    "elliptic_txs_edgelist.csv",
    "elliptic_txs_classes.csv",
)


def download_ssl_context() -> ssl.SSLContext:
    """Retain platform/custom trust roots and add certifi's public CA bundle."""
    context = ssl.create_default_context()
    context.load_verify_locations(cafile=certifi.where())
    return context


def fetch_archive(url: str, destination: Path) -> None:
    request = Request(url, headers={"User-Agent": "StreamGraphResearch/0.1.1"})
    try:
        with urlopen(request, timeout=90, context=download_ssl_context()) as response, destination.open("wb") as output:
            if response.status != 200:
                raise OSError(f"Unexpected HTTP status {response.status}")
            shutil.copyfileobj(response, output, length=2 * 1024 * 1024)
    except (URLError, ssl.SSLCertVerificationError) as error:
        reason = getattr(error, "reason", error)
        if isinstance(reason, ssl.SSLCertVerificationError):
            raise OSError(
                "HTTPS certificate verification failed. On python.org macOS installs, run "
                "the Install Certificates.command in /Applications/Python <version>/, "
                "then retry. StreamGraph keeps TLS verification enabled."
            ) from error
        raise OSError(f"Dataset download failed: {reason}") from error


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(content, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def download(raw_dir: Path, force: bool = False) -> dict:
    """Fetch only the named member of each ZIP; never extract arbitrary paths."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = raw_dir / "download_manifest.json"
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    entries = []
    for filename in FILES:
        target = raw_dir / filename
        url = f"{MIRROR}/{filename}.zip"
        if target.exists() and not force:
            previous = next((r for r in old.get("files", []) if r["filename"] == filename), None)
            if previous and sha256(target) != previous["sha256"]:
                raise ValueError(f"Raw file changed: {filename}; inspect it or download with --force")
            print(f"Using existing {filename}", flush=True)
        else:
            print(f"Downloading {filename} from the PyG-maintained mirror", flush=True)
            with tempfile.TemporaryDirectory(prefix="streamgraph-") as temporary_dir:
                archive_path = Path(temporary_dir) / "download.zip"
                fetch_archive(url, archive_path)
                with ZipFile(archive_path) as archive:
                    matching = [name for name in archive.namelist() if Path(name).name == filename]
                    if len(matching) != 1:
                        raise ValueError(f"Expected exactly one {filename} inside downloaded archive")
                    # A fixed destination plus exact member read avoids path traversal.
                    staging = target.with_suffix(".csv.part")
                    try:
                        with archive.open(matching[0]) as source, staging.open("wb") as output:
                            shutil.copyfileobj(source, output, length=2 * 1024 * 1024)
                        staging.replace(target)
                    finally:
                        staging.unlink(missing_ok=True)
        entries.append({"filename": filename, "source_url": url,
                        "size_bytes": target.stat().st_size, "sha256": sha256(target)})
        # Keep resumable provenance even if a later download fails.
        write_json(manifest_path, {"retrieved_at": utc_now(), "files": entries})
    return {"retrieved_at": utc_now(), "files": entries}


@dataclass
class Dataset:
    transaction_ids: np.ndarray
    time_steps: np.ndarray
    features: np.ndarray
    labels: np.ndarray
    edge_index: np.ndarray
    raw_hashes: dict

    def feature_view(self, name: str) -> np.ndarray:
        if name == "all":
            return self.features
        if name == "local":
            # 94 published local fields include the time step: 93 after removing it.
            return self.features[:, :93]
        raise ValueError(f"Unknown feature view: {name}")


def dataset_fingerprint(raw_hashes: dict) -> str:
    return hashlib.sha256(json.dumps(raw_hashes, sort_keys=True).encode()).hexdigest()


def load_dataset(raw_dir: Path, expected_release: bool = True) -> Dataset:
    for name in FILES:
        if not (raw_dir / name).is_file():
            raise FileNotFoundError(f"Missing {name}; run streamgraph download first")
    dtypes = {0: "int64", 1: "int16", **{i: "float32" for i in range(2, 167)}}
    features = pd.read_csv(raw_dir / FILES[0], header=None, dtype=dtypes)
    if features.shape[1] != 167:
        raise ValueError("Features must contain txId, time_step, and 165 numerical feature columns")
    ids = features.iloc[:, 0].to_numpy(dtype=np.int64, copy=True)
    steps = features.iloc[:, 1].to_numpy(dtype=np.int16, copy=True)
    x = features.iloc[:, 2:].to_numpy(dtype=np.float32, copy=True)
    if pd.Index(ids).has_duplicates:
        raise ValueError("Duplicate transaction IDs in features")
    if not np.isfinite(x).all():
        raise ValueError("Feature values must be finite; missing values require an explicit train-fitted policy")
    if not np.all((steps >= 1) & (steps <= 49)):
        raise ValueError("time_step must be in the original 1-49 range")

    classes = pd.read_csv(raw_dir / FILES[2], dtype={"txId": "int64", "class": "string"})
    if set(classes.columns) != {"txId", "class"}:
        raise ValueError("Class CSV must have txId and class columns")
    if classes["txId"].duplicated().any():
        raise ValueError("Duplicate class rows for a transaction ID")
    if len(classes) != len(ids) or set(classes["txId"]) != set(ids):
        raise ValueError("Feature and class transaction IDs do not match exactly")
    raw_labels = classes.set_index("txId")["class"].reindex(ids)
    mapping = {"unknown": -1, "1": 1, "2": 0}
    if not raw_labels.isin(mapping).all():
        raise ValueError("Unexpected class label; expected 1 (illicit), 2 (licit), or unknown")
    y = raw_labels.map(mapping).to_numpy(dtype=np.int8)

    edges = pd.read_csv(raw_dir / FILES[1], dtype={"txId1": "int64", "txId2": "int64"})
    if list(edges.columns) != ["txId1", "txId2"]:
        raise ValueError("Edge CSV must have txId1 and txId2 columns")
    index = pd.Index(ids)
    source = index.get_indexer(edges["txId1"])
    target = index.get_indexer(edges["txId2"])
    if (source < 0).any() or (target < 0).any():
        raise ValueError("A graph edge points to a transaction without features")
    if (steps[source] != steps[target]).any():
        raise ValueError("Unexpected cross-time-step edge; original Elliptic snapshots should be disconnected")
    edge_index = np.stack([source, target]).astype(np.int64)
    if expected_release and (len(ids), len(edges)) != (203769, 234355):
        raise ValueError("Unexpected release counts: expected 203,769 nodes and 234,355 edges")
    hashes = {name: sha256(raw_dir / name) for name in FILES}
    return Dataset(ids, steps, x, y, edge_index, hashes)


def label_counts(labels: np.ndarray) -> dict:
    return {"licit": int((labels == 0).sum()), "illicit": int((labels == 1).sum()),
            "unknown": int((labels == -1).sum())}


def audit(dataset: Dataset, config: dict) -> dict:
    masks = split_masks(dataset.time_steps, dataset.labels, config)
    per_step = []
    source_steps = dataset.time_steps[dataset.edge_index[0]]
    for step in range(1, 50):
        selected = dataset.time_steps == step
        per_step.append({"time_step": step, "nodes": int(selected.sum()),
                         "edges": int((source_steps == step).sum()),
                         **label_counts(dataset.labels[selected])})
    return {"created_at": utc_now(), "dataset": "Elliptic original release",
            "nodes": len(dataset.labels), "edges": dataset.edge_index.shape[1],
            "numeric_features_without_time": dataset.features.shape[1],
            "label_mapping": {"1": "illicit=1", "2": "licit=0", "unknown": -1},
            "labels": label_counts(dataset.labels), "cross_step_edges": 0,
            "raw_file_hashes": dataset.raw_hashes,
            "dataset_fingerprint": dataset_fingerprint(dataset.raw_hashes),
            "config_sha256": config_hash(config), "feature_view": config["feature_view"],
            "splits": {role: {"time_steps": config["splits"][role],
                              "nodes": int(masks[role].sum()),
                              "known_labels": int(masks[role + "_known"].sum()),
                              **label_counts(dataset.labels[masks[role]])} for role in ROLES},
            "per_time_step": per_step,
            "final_test_predictions_generated": False}


def save_prepared(dataset: Dataset, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, transaction_ids=dataset.transaction_ids,
                        time_steps=dataset.time_steps, features=dataset.features,
                        labels=dataset.labels, edge_index=dataset.edge_index)
    write_json(destination.with_suffix(".json"), {"raw_file_hashes": dataset.raw_hashes,
               "dataset_fingerprint": dataset_fingerprint(dataset.raw_hashes)})


def load_prepared(path: Path) -> Dataset:
    metadata = json.loads(path.with_suffix(".json").read_text())
    with np.load(path, allow_pickle=False) as content:
        dataset = Dataset(content["transaction_ids"], content["time_steps"],
                          content["features"], content["labels"], content["edge_index"],
                          metadata["raw_file_hashes"])
    return dataset


def export_snapshot(dataset: Dataset, step: int, output: Path, feature_view: str = "all") -> dict:
    if not 1 <= step <= 49:
        raise ValueError("Snapshot must be between 1 and 49")
    selected = np.flatnonzero(dataset.time_steps == step)
    old_to_new = np.full(len(dataset.labels), -1, dtype=np.int64)
    old_to_new[selected] = np.arange(len(selected))
    edge_mask = (dataset.time_steps[dataset.edge_index[0]] == step)
    local_edges = old_to_new[dataset.edge_index[:, edge_mask]]
    if local_edges.size and (local_edges < 0).any():
        raise ValueError("Snapshot contains an edge outside its node set")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, x=dataset.feature_view(feature_view)[selected],
                        y=dataset.labels[selected], edge_index=local_edges,
                        transaction_ids=dataset.transaction_ids[selected],
                        known_mask=dataset.labels[selected] >= 0, time_step=np.array(step))
    return {"time_step": step, "nodes": len(selected), "edges": local_edges.shape[1],
            "features": dataset.feature_view(feature_view).shape[1],
            "unknown_labels_preserved": True, "output": str(output)}
