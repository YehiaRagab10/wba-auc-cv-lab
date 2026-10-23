"""Shared tracking utilities for all WBA2026 model trainers.

Provides CSV metrics logging, git commit tracking, and full config
serialization for MLflow — replacing ad-hoc txt logging and partial
param logging in individual trainers.
"""

from __future__ import annotations

import csv
import os
import subprocess
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

MODELS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODELS_DIR.parent

load_dotenv(PROJECT_ROOT / ".env", override=True)

import mlflow

if os.environ.get("MLFLOW_TRACKING_URI"):
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])


def get_git_info() -> dict[str, str]:
    """Capture current git commit, branch, and dirty status."""
    info: dict[str, str] = {}
    try:
        info["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        info["git_branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        info["git_dirty"] = str(bool(dirty))
    except (subprocess.CalledProcessError, FileNotFoundError):
        info["git_commit"] = "unknown"
        info["git_branch"] = "unknown"
        info["git_dirty"] = "unknown"
    return info


def config_to_params(config: Any) -> dict[str, str]:
    """Serialize all fields of a frozen Settings dataclass to string params."""
    params: dict[str, str] = {}
    for f in fields(config):
        val = getattr(config, f.name)
        if isinstance(val, Path):
            try:
                val = val.relative_to(PROJECT_ROOT)
            except ValueError:
                pass
        params[f.name] = str(val)
    return params


def setup_mlflow_run(
    experiment_name: str,
    run_name: str,
    config: Any,
) -> None:
    """Initialize MLflow run with full config + git info."""
    mlflow.set_experiment(experiment_name)
    mlflow.start_run(run_name=run_name)
    mlflow.log_params(config_to_params(config))
    mlflow.log_params(get_git_info())


def register_model(
    model_name: str,
    checkpoint_path: Path,
    experiment_name: str | None = None,
    metrics: dict[str, float] | None = None,
    official_score: float | None = None,
) -> None:
    """Register a checkpoint as a new version in the MLflow Model Registry.

    Each call creates a new version under ``model_name`` (e.g. "stgcn",
    "nvidia-tcnn"). The team can then compare versions and promote the
    best one.

    Pass ``metrics`` (the full eval dict) and ``official_score`` so the
    registered model version is linked to its validation performance.
    This lets anyone browsing the registry see exactly how good the model is.

    Call this ONLY after evaluating on the full validation set and
    confirming the model is worth sharing.
    """
    if experiment_name:
        mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=f"{model_name}-register") as run:
        mlflow.log_params(get_git_info())
        if metrics:
            mlflow.log_metrics({f"val/{k}": v for k, v in metrics.items()})
        if official_score is not None:
            mlflow.log_metric("official_score", official_score)
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=mlflow.pyfunc.PythonModel(),
            artifacts={"checkpoint": str(checkpoint_path)},
            registered_model_name=model_name,
        )
        score_str = f" | score={official_score:.6f}" if official_score is not None else ""
        print(
            f"Registered '{model_name}' as new version "
            f"(source: {checkpoint_path.name}){score_str}"
        )


class CSVMetricsLogger:
    """Append-safe CSV logger that creates one file per metric group.

    Each call to ``log`` writes one row. The CSV is created fresh at the
    start of each training run (the ``output_dir`` is expected to be
    run-specific, so overwriting is fine).
    """

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._writers: dict[str, csv.DictWriter] = {}
        self._files: dict[str, Any] = {}

    def log(self, group: str, metrics: dict[str, float], step: int) -> None:
        row = {"step": step, **metrics}
        if group not in self._writers:
            path = self.output_dir / f"{group}_metrics.csv"
            fh = open(path, "w", newline="")
            writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
            writer.writeheader()
            self._writers[group] = writer
            self._files[group] = fh
        self._writers[group].writerow(row)
        self._files[group].flush()

    def close(self) -> None:
        for fh in self._files.values():
            fh.close()
        self._writers.clear()
        self._files.clear()
