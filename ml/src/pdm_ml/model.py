"""Load the trained models and predict for the latest cycle of each machine."""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import build


class Model:
    def __init__(self, path: Path):
        path = Path(path)
        self.meta = json.loads((path / "meta.json").read_text())
        self.classifier = lgb.Booster(model_file=str(path / "classifier.txt"))
        self.rul = lgb.Booster(model_file=str(path / "rul.txt"))
        self.sensors = self.meta["sensors"]
        self.features = self.meta["features"]
        self.alert_rul = self.meta["metrics"]["alert_rul_cycles"]

    @property
    def version(self) -> str:
        return self.meta["version"]

    def predict_latest(self, history: pd.DataFrame) -> list[dict]:
        """history: rows (unit, cycle, sensors...) for one or more machines; returns one prediction per machine."""
        feats = build(history, self.sensors)
        last = feats.groupby("unit").tail(1)
        prob = self.classifier.predict(last[self.features])
        rul = np.clip(self.rul.predict(last[self.features]), 0, None)
        out = []
        for unit, cycle, p, r in zip(last["unit"], last["cycle"], prob, rul, strict=True):
            out.append(
                {
                    "unit": unit,
                    "cycle": int(cycle),
                    "failure_probability": float(p),
                    "predicted_rul": float(r),
                    "health": float(np.clip(r / self.meta["rul_cap"], 0, 1) * 100),
                    "alert": bool(r <= self.alert_rul),
                }
            )
        return out
