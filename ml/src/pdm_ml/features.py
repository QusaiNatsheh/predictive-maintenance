"""Causal rolling features: each row only uses that engine's current and earlier cycles.

The same function serves training (whole histories) and the live simulator (history so far), so a prediction made
in the middle of a replay equals the one made offline for that cycle.
"""

import numpy as np
import pandas as pd

WINDOWS = (5, 10, 20)


def _slope(values: np.ndarray) -> float:
    n = len(values)
    if n < 3:
        return 0.0
    x = np.arange(n, dtype=float)
    x -= x.mean()
    return float((x * (values - values.mean())).sum() / (x * x).sum())


def _baseline(x: pd.Series) -> pd.Series:
    """Mean of the engine's first 5 cycles, using only cycles seen so far (causal for cycles 1-4 as well)."""
    first = x.expanding().mean()
    first[np.arange(len(x)) >= 5] = np.nan
    return first.ffill()


def build(df: pd.DataFrame, sensors: list[str]) -> pd.DataFrame:
    """One feature row per input row (sorted by unit and cycle)."""
    df = df.sort_values(["unit", "cycle"]).reset_index(drop=True)
    cols = {"unit": df["unit"], "cycle": df["cycle"]}
    g = df.groupby("unit", sort=False)
    for s in sensors:
        cols[s] = df[s]
        for w in WINDOWS:
            roll = g[s].rolling(w, min_periods=1)
            cols[f"{s}_mean{w}"] = roll.mean().reset_index(level=0, drop=True)
            cols[f"{s}_std{w}"] = roll.std().reset_index(level=0, drop=True).fillna(0.0)
        cols[f"{s}_slope20"] = (
            g[s].rolling(20, min_periods=3).apply(_slope, raw=True).reset_index(level=0, drop=True).fillna(0.0)
        )
        # drift from this engine's own starting level (engines start with different wear)
        cols[f"{s}_drift"] = cols[f"{s}_mean10"] - g[s].transform(_baseline)
    return pd.DataFrame(cols)


def feature_names(sensors: list[str]) -> list[str]:
    names = ["cycle"]
    for s in sensors:
        names.append(s)
        for w in WINDOWS:
            names += [f"{s}_mean{w}", f"{s}_std{w}"]
        names += [f"{s}_slope20", f"{s}_drift"]
    return names
