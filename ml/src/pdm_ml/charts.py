"""README charts: cost of the maintenance rule by trigger level, and one held-out engine replayed to failure."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE, TEXT, MUTED, GRID = "#fcfcfb", "#2b2b29", "#6b6b66", "#e6e5e0"
BLUE, ORANGE = "#2a78d6", "#eb6834"  # validated categorical slots 1 and 2


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def cost_curve(meta: dict, path: Path) -> None:
    curve = meta["metrics"]["rule_curve"]
    k = [c["k"] for c in curve]
    fig, ax = plt.subplots(figsize=(6.4, 4), facecolor=SURFACE)
    _style(ax)
    ax.plot(k, [c["cost_per_engine"] for c in curve], color=BLUE, linewidth=2)
    best = meta["metrics"]["alert_rul_cycles"]
    best_cost = next(c["cost_per_engine"] for c in curve if c["k"] == best)
    ax.scatter([best], [best_cost], s=60, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.annotate(
        f"chosen: plan maintenance at\npredicted life <= {best} cycles",
        (best, best_cost),
        xytext=(10, 60),
        textcoords="offset points",
        color=TEXT,
        fontsize=8,
    )
    ax.set_ylim(0, 4)
    ax.set_xlabel("Trigger: predicted remaining life (cycles)", color=MUTED, fontsize=9)
    ax.set_ylabel("Cost per engine (planned maintenance = 1)", color=MUTED, fontsize=9)
    ax.set_title("Choosing when to act (80 training engines, out-of-fold)", color=TEXT, fontsize=12, loc="left")
    ax.text(13, 3.6, "too late: failures before maintenance", color=MUTED, fontsize=8)
    ax.text(40, 0.85, "too early: useful life thrown away", color=MUTED, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def engine_replay(preds: pd.DataFrame, meta: dict, path: Path) -> None:
    unit = int(preds.groupby("unit")["cycle"].max().idxmax())  # the longest held-out engine
    e = preds[preds["unit"] == unit].sort_values("cycle")
    life = e["cycle"].max()
    k = meta["metrics"]["alert_rul_cycles"]
    alert = e[e["predicted_rul"] <= k]["cycle"].min()
    fig, ax = plt.subplots(figsize=(6.8, 4), facecolor=SURFACE)
    _style(ax)
    ax.plot(
        e["cycle"],
        e["rul"].clip(upper=meta["rul_cap"]),
        color=MUTED,
        linewidth=1.5,
        linestyle="--",
        label="true remaining life (capped at 125)",
    )
    ax.plot(e["cycle"], e["predicted_rul"], color=BLUE, linewidth=2, label="predicted remaining life")
    ax.axvline(alert, color=ORANGE, linewidth=2)
    ax.text(
        alert - 3,
        108,
        f"maintenance request\nat cycle {alert}\n({life - alert} cycles before failure)",
        color=TEXT,
        fontsize=8,
        ha="right",
    )
    ax.axvline(life, color=TEXT, linewidth=1)
    ax.text(life + 2, 60, "failure", color=TEXT, fontsize=8, rotation=90, va="center")
    ax.set_xlabel("Operating cycle", color=MUTED, fontsize=9)
    ax.set_ylabel("Remaining life (cycles)", color=MUTED, fontsize=9)
    ax.set_title(f"Held-out engine {unit}, replayed as if live", color=TEXT, fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main(model_dir: str = "models/latest", out: str = "../docs") -> None:
    model_dir, out = Path(model_dir), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((model_dir / "meta.json").read_text())
    cost_curve(meta, out / "cost_curve.png")
    engine_replay(pd.read_parquet(model_dir / "holdout_predictions.parquet"), meta, out / "engine_replay.png")
    print(f"charts -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
