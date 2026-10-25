import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from pdm_ml import service
from pdm_ml.data import WINDOW, informative_sensors, load
from pdm_ml.features import build
from pdm_ml.train import LEAD_TIME, policy_cost


def test_labels(raw):
    train, test = load(raw / "cmapss")
    last = train.groupby("unit").tail(1)
    assert (last["rul"] == 0).all()  # training engines run to failure
    assert (test.groupby("unit")["rul"].min() > 0).all()  # test engines stop before failure
    assert ((train["rul"] <= WINDOW) == train["fail_soon"].astype(bool)).all()


def test_features_use_only_the_past(raw):
    train, _ = load(raw / "cmapss")
    sensors = informative_sensors(train)
    engine = train[train["unit"] == 7]
    full = build(engine, sensors)
    for cut in (3, 40):
        partial = build(engine[engine["cycle"] <= cut], sensors)
        a = full.iloc[cut - 1].drop("unit").to_numpy(float)
        b = partial.iloc[cut - 1].drop("unit").to_numpy(float)
        assert np.allclose(a, b)


def test_policy_cost_counts_late_alerts_as_failures():
    life = {1: 200, 2: 200}
    costs = policy_cost(life, {1: 150, 2: 200 - LEAD_TIME + 1}, mean_life=200)
    assert costs["missed_failures"] == 1
    assert np.isclose(costs["cost_per_engine"], (1 + 50 / 200 + 10) / 2)


def test_model_beats_baselines(trained):
    _, meta = trained
    m = meta["metrics"]
    assert m["test_rul_last_cycle"]["model_rmse"] < m["test_rul_last_cycle"]["fleet_average_rmse"]
    assert m["test_warning"]["model"]["pr_auc"] > m["test_warning"]["health_indicator"]["pr_auc"]
    policy = m["holdout_policy"]
    model_cost = next(v["cost_per_engine"] for k, v in policy.items() if k.startswith("model"))
    assert model_cost < policy["reactive (run to failure)"]["cost_per_engine"]
    assert model_cost < next(v["cost_per_engine"] for k, v in policy.items() if k.startswith("fixed"))


def test_service_replay_matches_offline(trained, raw, monkeypatch):
    path, meta = trained
    monkeypatch.setenv("MODEL_DIR", str(path))
    monkeypatch.setenv("RAW_DIR", str(raw / "cmapss"))
    service.get_model.cache_clear()
    service.get_fleet.cache_clear()
    client = TestClient(service.app)
    client.post("/simulator/reset", json={"cycles_before_failure": 40})
    refs = ",".join(m["ref"] for m in client.get("/machines").json())
    first = {}
    for _ in range(41):
        for p in client.get("/predict", params={"refs": refs}).json():
            if p["alert"] and not p["failed"]:
                first.setdefault(p["ref"], p["cycle"])
        client.post("/simulator/tick", json={"cycles": 1})
    preds = pd.read_parquet(path / "holdout_predictions.parquet")
    k = meta["metrics"]["alert_rul_cycles"]
    for unit, g in preds.groupby("unit"):
        offline = g[g["predicted_rul"] <= k]["cycle"]
        expected = max(int(offline.min()), int(g["cycle"].max()) - 40) if len(offline) else None
        assert first.get(f"ENG-{int(unit):03d}") == expected
    assert client.get("/predict", params={"refs": "NOPE"}).status_code == 404


def test_readings_endpoint(trained, raw, monkeypatch):
    path, _ = trained
    monkeypatch.setenv("MODEL_DIR", str(path))
    service.get_model.cache_clear()
    train, _ = load(raw / "cmapss")
    engine = train[train["unit"] == 5]
    sensors = service.get_model().sensors
    readings = [{"cycle": int(r["cycle"]), "values": {s: float(r[s]) for s in sensors}} for _, r in engine.iterrows()]
    client = TestClient(service.app)
    res = client.post("/predict/readings", json={"ref": "PUMP-1", "readings": readings})
    assert res.status_code == 200 and res.json()["predicted_rul"] < 30  # last cycle is the failure
    bad = {"ref": "X", "readings": [{"cycle": 1, "values": {"s2": 1.0}}]}
    assert client.post("/predict/readings", json=bad).status_code == 422


def test_ai4i_never_uses_failure_modes(raw):
    from pdm_ml import ai4i

    out = ai4i.main(str(raw / "ai4i" / "ai4i2020.csv"))
    assert out["documented_rules"]["precision"] == 1.0
    assert out["model_with_physics_features"]["roc_auc"] > 0.9
