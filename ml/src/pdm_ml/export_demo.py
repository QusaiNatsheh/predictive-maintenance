"""Facts about the held-out engines for the Odoo demo check: failure cycle and the offline first-alert cycle.
Usage: python -m pdm_ml.export_demo --model models/latest --out data/generated/demo.json"""

import argparse
import json
from pathlib import Path

import pandas as pd


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/latest"))
    parser.add_argument("--out", type=Path, default=Path("data/generated/demo.json"))
    args = parser.parse_args(argv)
    meta = json.loads((args.model / "meta.json").read_text())
    preds = pd.read_parquet(args.model / "holdout_predictions.parquet")
    k = meta["metrics"]["alert_rul_cycles"]
    engines = []
    for unit, g in preds.groupby("unit"):
        alerts = g[g["predicted_rul"] <= k]["cycle"]
        engines.append(
            {
                "ref": f"ENG-{int(unit):03d}",
                "failure_cycle": int(g["cycle"].max()),
                "offline_first_alert": int(alerts.min()) if len(alerts) else None,
            }
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {"alert_rule_cycles": k, "lead_time": meta["costs"]["lead_time_cycles"], "engines": engines}, indent=1
        )
    )
    print(f"{len(engines)} engines -> {args.out}")


if __name__ == "__main__":
    main()
