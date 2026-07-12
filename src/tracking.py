from __future__ import annotations

import os
import mlflow

REGISTERED_MODEL_NAME = "walmart-wmae-champion"


def setup_mlflow(experiment_name: str) -> str:
    """Point MLflow at the configured backend and select the experiment."""
    uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)
    return uri
