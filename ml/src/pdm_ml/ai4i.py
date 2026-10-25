"""Cross-check on the UCI AI4I 2020 dataset (CC BY 4.0): one row per process, label "Machine failure".

The failure-mode columns (TWF, HDF, PWF, OSF, RNF) define the label, so they are never used as features.
The dataset is itself synthetic: its documentation states the rules that generate failures. A "documented rules"
baseline applies those rules directly (tool-wear failures and random failures are random by design).
Usage: python -m pdm_ml.ai4i data/raw/ai4i/ai4i2020.csv
"""

import json
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

OSF_LIMIT = {"L": 11000, "M": 12000, "H": 13000}


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.split(" [")[0].lower().replace(" ", "_") for c in df.columns]
    df["type"] = df["type"].astype("category")
    df["power_w"] = df["torque"] * df["rotational_speed"] * 2 * np.pi / 60
    df["temp_diff"] = df["process_temperature"] - df["air_temperature"]
    df["wear_x_torque"] = df["tool_wear"] * df["torque"]
    return df


def documented_rules(df: pd.DataFrame) -> np.ndarray:
    hdf = (df["temp_diff"] < 8.6) & (df["rotational_speed"] < 1380)
    pwf = (df["power_w"] < 3500) | (df["power_w"] > 9000)
    osf = df["wear_x_torque"] > df["type"].astype(str).map(OSF_LIMIT)
    return (hdf | pwf | osf).astype(float).to_numpy()


def main(path: str) -> dict:
    df = load(path)
    y = df["machine_failure"].to_numpy()
    raw = ["type", "air_temperature", "process_temperature", "rotational_speed", "torque", "tool_wear"]
    out = {"rows": len(df), "failure_rate": float(y.mean())}
    for name, cols in (
        ("model_raw_sensors", raw),
        ("model_with_physics_features", raw + ["power_w", "temp_diff", "wear_x_torque"]),
    ):
        p = np.zeros(len(df))
        for fit, val in StratifiedKFold(5, shuffle=True, random_state=0).split(df, y):
            b = lgb.train(
                {"objective": "binary", "learning_rate": 0.05, "num_leaves": 15, "verbose": -1, "seed": 1},
                lgb.Dataset(df.iloc[fit][cols], y[fit]),
                300,
            )
            p[val] = b.predict(df.iloc[val][cols])
        out[name] = {"roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p))}
    rules = documented_rules(df)
    out["documented_rules"] = {
        "roc_auc": float(roc_auc_score(y, rules)),
        "pr_auc": float(average_precision_score(y, rules)),
        "recall": float(rules[y == 1].mean()),
        "precision": float(y[rules == 1].mean()),
    }
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/raw/ai4i/ai4i2020.csv")
