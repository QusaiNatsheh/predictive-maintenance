"""Prediction service with a built-in sensor replay. Run: uvicorn pdm_ml.service:app

- GET  /machines                 simulated machines, their current cycle and whether they have failed
- POST /simulator/reset, /tick   control the replay (demo)
- GET  /predict?refs=...         prediction for simulated machines at their current cycle
- POST /predict/readings         prediction from raw readings sent by the caller (real sensors)
"""

import os
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from .data import load
from .model import Model
from .simulator import Fleet


@lru_cache
def get_model() -> Model:
    return Model(Path(os.environ.get("MODEL_DIR", "models/latest")))


@lru_cache
def get_fleet() -> Fleet:
    model = get_model()
    train, _ = load(Path(os.environ.get("RAW_DIR", "data/raw/cmapss")))
    fleet = Fleet.from_frame(train, model.meta["metrics"]["split"]["holdout_engines"])
    fleet.reset(int(os.environ.get("SIM_START_BEFORE_FAILURE", "60")))
    return fleet


class Prediction(BaseModel):
    ref: str
    cycle: int
    failure_probability: float
    predicted_rul: float
    health: float
    alert: bool
    failed: bool = False


class ResetIn(BaseModel):
    cycles_before_failure: int | None = Field(60, ge=1, le=400)
    staggered: bool = False


class TickIn(BaseModel):
    cycles: int = Field(1, ge=1, le=100)


class Reading(BaseModel):
    cycle: int = Field(ge=1)
    values: dict[str, float]


class ReadingsIn(BaseModel):
    ref: str = Field(max_length=100)
    readings: list[Reading] = Field(min_length=1, max_length=5000)


app = FastAPI(title="Predictive maintenance service", version="1.0.0")


@app.get("/health")
def health(model: Model = Depends(get_model)):
    return {"status": "ok", "model_version": model.version, "alert_rule": f"predicted life <= {model.alert_rul}"}


@app.get("/machines")
def machines(fleet: Fleet = Depends(get_fleet)):
    return [{"ref": m.ref, "cycle": m.position, "failed": m.failed} for m in fleet.machines.values()]


@app.post("/simulator/reset")
def reset(body: ResetIn, fleet: Fleet = Depends(get_fleet)):
    fleet.reset(body.cycles_before_failure, staggered=body.staggered)
    return {"status": "reset", "machines": len(fleet.machines)}


@app.post("/simulator/tick")
def tick(body: TickIn, fleet: Fleet = Depends(get_fleet)):
    fleet.tick(body.cycles)
    return {"failed": [m.ref for m in fleet.machines.values() if m.failed]}


@app.get("/predict", response_model=list[Prediction])
def predict(
    refs: str = Query(..., max_length=5000), model: Model = Depends(get_model), fleet: Fleet = Depends(get_fleet)
):
    wanted = [r.strip() for r in refs.split(",") if r.strip()]
    unknown = [r for r in wanted if r not in fleet.machines]
    if unknown:
        raise HTTPException(status_code=404, detail=f"Unknown machines: {', '.join(unknown)}")
    history = pd.concat([fleet.machines[r].history() for r in wanted])
    preds = model.predict_latest(history)
    return [
        Prediction(
            ref=p["unit"], failed=fleet.machines[p["unit"]].failed, **{k: v for k, v in p.items() if k != "unit"}
        )
        for p in preds
    ]


@app.post("/predict/readings", response_model=Prediction)
def predict_readings(body: ReadingsIn, model: Model = Depends(get_model)):
    missing = [s for s in model.sensors if any(s not in r.values for r in body.readings)]
    if missing:
        raise HTTPException(status_code=422, detail=f"Every reading needs sensors: {', '.join(sorted(set(missing)))}")
    rows = [{"unit": body.ref, "cycle": r.cycle, **r.values} for r in body.readings]
    p = model.predict_latest(pd.DataFrame(rows))[0]
    return Prediction(ref=body.ref, **{k: v for k, v in p.items() if k != "unit"})
