"""Validate explicit time ranges before any fitting occurs."""

import hashlib
import json
from pathlib import Path

import numpy as np


ROLES = ("source", "adaptation", "validation", "test")


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    if config.get("feature_view") not in {"all", "local"}:
        raise ValueError("feature_view must be all or local")
    previous_end = 0
    for role in ROLES:
        bounds = config.get("splits", {}).get(role)
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ValueError(f"Missing [start, end] time range for {role}")
        start, end = bounds
        if type(start) is not int or type(end) is not int:
            raise ValueError("Time ranges must contain integers")
        if start != previous_end + 1 or end < start:
            raise ValueError("Time ranges must be chronological, contiguous, and disjoint")
        previous_end = end
    if previous_end != 49:
        raise ValueError("The four roles must cover original Elliptic steps 1-49")
    if type(config.get("seed")) is not int:
        raise ValueError("seed must be an integer")


def config_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def split_masks(time_steps: np.ndarray, labels: np.ndarray, config: dict) -> dict:
    """Keep graph membership separate from eligibility for supervised fitting."""
    validate_config(config)
    masks = {}
    for role in ROLES:
        start, end = config["splits"][role]
        member = (time_steps >= start) & (time_steps <= end)
        masks[role] = member
        masks[role + "_known"] = member & (labels >= 0)
    # Only source labels are available to the baseline training stage.
    masks["train"] = masks["source_known"]
    return masks
