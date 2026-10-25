"""NASA C-MAPSS turbofan data (Saxena & Goebel, 2008): loading and labels.

Each engine (unit) is a time series of operating cycles. Training engines run until failure; test engines stop
earlier, and RUL_FDxxx.txt gives their true remaining life after the last recorded cycle.
"""

from pathlib import Path

import pandas as pd

SETTINGS = ["setting_1", "setting_2", "setting_3"]
SENSORS = [f"s{i}" for i in range(1, 22)]
COLUMNS = ["unit", "cycle", *SETTINGS, *SENSORS]
WINDOW = 30  # "fails within the next 30 cycles" is the alert target
RUL_CAP = 125  # usual cap in the literature: early in life the exact remaining life is not observable


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.iloc[:, : len(COLUMNS)]
    df.columns = COLUMNS
    return df


def load(raw_dir: Path, subset: str = "FD001") -> tuple[pd.DataFrame, pd.DataFrame]:
    """(train, test), each with `rul` (true remaining cycles) and `fail_soon` (rul <= WINDOW)."""
    raw_dir = Path(raw_dir)
    train = _read(raw_dir / f"train_{subset}.txt")
    train["rul"] = train.groupby("unit")["cycle"].transform("max") - train["cycle"]
    test = _read(raw_dir / f"test_{subset}.txt")
    final_rul = pd.read_csv(raw_dir / f"RUL_{subset}.txt", header=None).iloc[:, 0]
    last = test.groupby("unit")["cycle"].transform("max")
    test["rul"] = (
        test["unit"].map(dict(zip(range(1, len(final_rul) + 1), final_rul, strict=True))) + last - test["cycle"]
    )
    for df in (train, test):
        df["fail_soon"] = (df["rul"] <= WINDOW).astype(int)
    return train, test


def informative_sensors(train: pd.DataFrame, min_std: float = 1e-3) -> list[str]:
    """Sensors that vary in the training data (several are constant under a single operating condition)."""
    return [s for s in SENSORS if train[s].std() > min_std]
