import os
from pathlib import Path

import pytest

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


@pytest.fixture(scope="session")
def raw():
    if not (RAW / "cmapss" / "train_FD001.txt").exists():
        pytest.skip("datasets missing: run ./download_data.sh data/raw")
    return RAW


@pytest.fixture(scope="session")
def trained(raw, tmp_path_factory):
    from pdm_ml.train import train

    os.environ["MLFLOW_TRACKING_URI"] = f"sqlite:///{tmp_path_factory.mktemp('mlflow')}/m.db"
    out = tmp_path_factory.mktemp("model")
    return out, train(raw / "cmapss", out)
