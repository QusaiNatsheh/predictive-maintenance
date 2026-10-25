"""Train the failure-warning and remaining-life models, choose the maintenance rule by cost, and evaluate.

Engines are never split across groups:
- 80 training engines: models, cross-validated by engine (out-of-fold predictions choose the maintenance rule)
- 20 held-out engines (run to failure, never used for training or tuning): maintenance policy evaluation and
  the Odoo demo replay
- NASA's 100 test engines (stop before failure, true remaining life given): the standard benchmark

Usage: python -m pdm_ml.train --raw data/raw/cmapss --out models/latest
"""

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from .data import RUL_CAP, WINDOW, informative_sensors, load
from .features import build, feature_names

CLS_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
    "seed": 7,
}
REG_PARAMS = {**CLS_PARAMS, "objective": "regression"}
ROUNDS = 400
HOLDOUT_SEED = 13
# Cost assumptions (relative units, stated in the README):
COST_FAILURE = 10.0  # unplanned failure: repair plus downtime
COST_MAINTENANCE = 1.0  # planned maintenance
COST_WASTED_LIFE = 1.0  # value of one full engine life thrown away by replacing too early
LEAD_TIME = 10  # cycles of notice needed to schedule maintenance; a later warning cannot prevent the failure


def split_units(units: np.ndarray, n_holdout: int = 20) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(HOLDOUT_SEED)
    holdout = np.sort(rng.choice(units, size=n_holdout, replace=False))
    return np.setdiff1d(units, holdout), holdout


def first_alerts(units: np.ndarray, cycles: np.ndarray, alert: np.ndarray) -> dict:
    """First cycle at which each unit raises an alert (None if never)."""
    out = {}
    for u in np.unique(units):
        m = (units == u) & alert
        out[int(u)] = int(cycles[m].min()) if m.any() else None
    return out


def policy_cost(life: dict[int, int], alert_cycle: dict[int, int | None], mean_life: float) -> dict:
    """Maintain at the first alert if it gives at least LEAD_TIME cycles of notice; otherwise the engine fails.

    Life wasted by maintaining early has a cost proportional to the share of an average life thrown away.
    """
    costs, warnings, missed = [], [], 0
    for u, failure in life.items():
        a = alert_cycle.get(u)
        if a is None or failure - a < LEAD_TIME:
            costs.append(COST_FAILURE)
            missed += 1
        else:
            remaining = failure - a
            warnings.append(remaining)
            costs.append(COST_MAINTENANCE + COST_WASTED_LIFE * remaining / mean_life)
    return {
        "cost_per_engine": float(np.mean(costs)),
        "missed_failures": missed,
        "engines": len(life),
        "median_warning_cycles": float(np.median(warnings)) if warnings else None,
        "min_warning_cycles": float(np.min(warnings)) if warnings else None,
    }


def classification(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    alert = p >= threshold
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "precision": float(precision_score(y, alert, zero_division=0)),
        "recall": float(recall_score(y, alert, zero_division=0)),
        "n": int(len(y)),
        "positives": int(y.sum()),
    }


def train(raw: Path, out: Path) -> dict:
    train_df, test_df = load(raw)
    sensors = informative_sensors(train_df)
    names = feature_names(sensors)
    feats = build(train_df, sensors).assign(
        rul=train_df.sort_values(["unit", "cycle"])["rul"].to_numpy(),
        fail_soon=train_df.sort_values(["unit", "cycle"])["fail_soon"].to_numpy(),
    )
    train_units, holdout_units = split_units(feats["unit"].unique())
    tr = feats[feats["unit"].isin(train_units)].reset_index(drop=True)
    ho = feats[feats["unit"].isin(holdout_units)].reset_index(drop=True)
    life_tr = tr.groupby("unit")["cycle"].max().to_dict()
    life_ho = ho.groupby("unit")["cycle"].max().to_dict()
    mean_life = float(np.mean(list(life_tr.values())))

    # Out-of-fold predictions on the 80 training engines (grouped by engine) choose the maintenance rule:
    # "plan maintenance when the predicted remaining life is at most k cycles".
    oof_rul = np.zeros(len(tr))
    for fit_idx, val_idx in GroupKFold(n_splits=5).split(tr, groups=tr["unit"]):
        b = lgb.train(
            REG_PARAMS, lgb.Dataset(tr.loc[fit_idx, names], tr.loc[fit_idx, "rul"].clip(upper=RUL_CAP)), ROUNDS
        )
        oof_rul[val_idx] = b.predict(tr.loc[val_idx, names])
    units_tr, cycles_tr = tr["unit"].to_numpy(), tr["cycle"].to_numpy()
    curve = [
        {"k": k, **policy_cost(life_tr, first_alerts(units_tr, cycles_tr, oof_rul <= k), mean_life)}
        for k in range(5, 61, 1)
    ]
    best = min(curve, key=lambda c: c["cost_per_engine"])
    k_alert = best["k"]

    cls = lgb.train(CLS_PARAMS, lgb.Dataset(tr[names], tr["fail_soon"]), ROUNDS)
    reg = lgb.train(REG_PARAMS, lgb.Dataset(tr[names], tr["rul"].clip(upper=RUL_CAP)), ROUNDS)

    # Baselines, built from the training engines only
    fixed_grid = range(80, 400, 2)
    fixed_costs = [
        (t, policy_cost(life_tr, dict.fromkeys(life_tr, t), mean_life)["cost_per_engine"]) for t in fixed_grid
    ]
    fixed_t = min(fixed_costs, key=lambda c: c[1])[0]
    hi_sensor, hi_threshold, hi_sign = _health_indicator(tr, sensors)

    def hi_alert(df):
        return hi_sign * df[f"{hi_sensor}_drift"].to_numpy() >= hi_threshold

    results = {
        "alert_rul_cycles": k_alert,
        "mean_life_train": mean_life,
        "fixed_interval_cycles": fixed_t,
        "health_indicator": {"feature": f"{hi_sensor}_drift", "threshold": hi_threshold, "sign": hi_sign},
        "rule_curve": curve,
    }
    # Held-out engines: maintenance policies
    hp = cls.predict(ho[names])
    h_rul = reg.predict(ho[names])
    units, cycles = ho["unit"].to_numpy(), ho["cycle"].to_numpy()
    results["holdout_policy"] = {
        "reactive (run to failure)": policy_cost(life_ho, dict.fromkeys(life_ho), mean_life),
        f"fixed interval ({fixed_t} cycles)": policy_cost(life_ho, dict.fromkeys(life_ho, fixed_t), mean_life),
        "health indicator threshold": policy_cost(life_ho, first_alerts(units, cycles, hi_alert(ho)), mean_life),
        f"model (predicted life <= {k_alert} cycles)": policy_cost(
            life_ho, first_alerts(units, cycles, h_rul <= k_alert), mean_life
        ),
    }
    # NASA test set: per-cycle warnings and remaining-life benchmark (last cycle of each engine)
    te = build(test_df, sensors).assign(
        rul=test_df.sort_values(["unit", "cycle"])["rul"].to_numpy(),
        fail_soon=test_df.sort_values(["unit", "cycle"])["fail_soon"].to_numpy(),
    )
    tp = cls.predict(te[names])
    fleet_p = (te["cycle"] >= mean_life - WINDOW).astype(float).to_numpy()
    results["test_warning"] = {
        "model": classification(te["fail_soon"].to_numpy(), tp, 0.5),
        "fleet_average_life": classification(te["fail_soon"].to_numpy(), fleet_p, 0.5),
        "health_indicator": classification(te["fail_soon"].to_numpy(), hi_alert(te).astype(float), 0.5),
    }
    last = te.groupby("unit").tail(1)
    rul_pred = np.clip(reg.predict(last[names]), 0, None)
    fleet_rul = np.clip(mean_life - last["cycle"].to_numpy(), 0, RUL_CAP)
    truth = last["rul"].to_numpy()
    results["test_rul_last_cycle"] = {
        "model_rmse": float(np.sqrt(np.mean((rul_pred - truth) ** 2))),
        "fleet_average_rmse": float(np.sqrt(np.mean((fleet_rul - truth) ** 2))),
        "model_mae": float(np.mean(np.abs(rul_pred - truth))),
        "engines": int(len(truth)),
    }
    results["split"] = {
        "train_engines": len(train_units),
        "holdout_engines": [int(u) for u in holdout_units],
        "test_engines": int(te["unit"].nunique()),
    }

    gain = cls.feature_importance("gain")
    importance = dict(sorted(zip(names, (gain / gain.sum()).round(4).tolist(), strict=True), key=lambda kv: -kv[1]))
    out.mkdir(parents=True, exist_ok=True)
    cls.save_model(str(out / "classifier.txt"))
    reg.save_model(str(out / "rul.txt"))
    meta = {
        "version": datetime.now(UTC).strftime("%Y%m%d%H%M%S"),
        "sensors": sensors,
        "features": names,
        "window": WINDOW,
        "rul_cap": RUL_CAP,
        "params": CLS_PARAMS,
        "rounds": ROUNDS,
        "costs": {
            "failure": COST_FAILURE,
            "maintenance": COST_MAINTENANCE,
            "wasted_life": COST_WASTED_LIFE,
            "lead_time_cycles": LEAD_TIME,
        },
        "metrics": results,
        "feature_importance_gain": dict(list(importance.items())[:20]),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    ho.assign(probability=hp, predicted_rul=h_rul).to_parquet(out / "holdout_predictions.parquet")
    _log_mlflow(meta, out)
    return meta


def _health_indicator(tr: pd.DataFrame, sensors: list[str]) -> tuple[str, float, int]:
    """Best single drift feature and threshold for the warning target (a typical rule-of-thumb monitor)."""
    best = (0.0, sensors[0], 0.0, 1)
    y = tr["fail_soon"].to_numpy()
    for s in sensors:
        x = tr[f"{s}_drift"].to_numpy()
        sign = 1 if np.corrcoef(x, y)[0, 1] >= 0 else -1
        for q in np.quantile(sign * x, np.linspace(0.5, 0.99, 50)):
            alert = sign * x >= q
            p, r = (y[alert].mean() if alert.any() else 0), y[alert].sum() / y.sum()
            f1 = 2 * p * r / (p + r) if p + r else 0
            if f1 > best[0]:
                best = (f1, s, float(q), sign)
    return best[1], best[2], best[3]


def _log_mlflow(meta: dict, out: Path) -> None:
    try:
        import mlflow
        import mlflow.lightgbm
    except ImportError:
        return
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment("predictive-maintenance")
    with mlflow.start_run(run_name=meta["version"]):
        m = meta["metrics"]
        mlflow.log_param("alert_rul_cycles", m["alert_rul_cycles"])
        for who, v in m["test_warning"].items():
            for k in ("roc_auc", "pr_auc", "precision", "recall"):
                mlflow.log_metric(f"test_{who}_{k}", v[k])
        mlflow.log_metric("test_rul_rmse", m["test_rul_last_cycle"]["model_rmse"])
        for who, v in m["holdout_policy"].items():
            mlflow.log_metric(f"holdout_cost_{who.split(' ')[0]}", v["cost_per_engine"])
        mlflow.log_artifact(str(out / "meta.json"))
        mlflow.lightgbm.log_model(
            lgb.Booster(model_file=str(out / "classifier.txt")),
            name="classifier",
            registered_model_name="engine-failure-warning",
        )


def print_report(meta: dict) -> None:
    m = meta["metrics"]
    print(
        f"Maintenance rule (lowest cost on out-of-fold training engines): predicted life <= "
        f"{m['alert_rul_cycles']} cycles"
    )
    print(f"\nNASA test set, warning 'fails within {meta['window']} cycles' (per cycle):")
    print(f"{'':22}{'ROC-AUC':>9}{'PR-AUC':>9}{'Precision':>11}{'Recall':>8}")
    for who, v in m["test_warning"].items():
        print(f"{who:22}{v['roc_auc']:9.3f}{v['pr_auc']:9.3f}{v['precision']:11.3f}{v['recall']:8.3f}")
    r = m["test_rul_last_cycle"]
    print(
        f"\nRemaining life at the last cycle (100 engines): model RMSE {r['model_rmse']:.1f} cycles "
        f"(MAE {r['model_mae']:.1f}), fleet-average baseline RMSE {r['fleet_average_rmse']:.1f}"
    )
    print(f"\nHeld-out engines ({len(m['split']['holdout_engines'])}, run to failure), maintenance policy:")
    print(f"{'':32}{'cost/engine':>12}{'missed':>8}{'median warning':>16}")
    for who, v in m["holdout_policy"].items():
        w = f"{v['median_warning_cycles']:.0f}" if v["median_warning_cycles"] is not None else "-"
        print(f"{who:32}{v['cost_per_engine']:12.2f}{v['missed_failures']:8}{w:>16}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/cmapss"))
    parser.add_argument("--out", type=Path, default=Path("models/latest"))
    args = parser.parse_args(argv)
    print_report(train(args.raw, args.out))


if __name__ == "__main__":
    main()
