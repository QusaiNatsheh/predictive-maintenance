"""Replay the held-out engines cycle by cycle through the service and Odoo; check that every engine gets a
maintenance request before it fails, with enough notice. Run inside `odoo shell` after loading the demo."""

import json
import os

import requests

data = json.load(open(os.environ.get("DEMO_JSON", "/mnt/data/demo.json"), encoding="utf-8"))
url = env["maintenance.equipment"]._pdm_service_url()
START = 60  # the replay starts this many cycles before each engine's (hidden) failure
requests.post(f"{url}/simulator/reset", json={"cycles_before_failure": START}, timeout=30).raise_for_status()
Equipment = env["maintenance.equipment"]
Request = env["maintenance.request"]
Request.search([("pdm_auto", "=", True)]).unlink()
for _ in range(START + 1):
    Equipment.search([("pdm_machine_ref", "!=", False)])._pdm_update()
    requests.post(f"{url}/simulator/tick", json={"cycles": 1}, timeout=30).raise_for_status()
env.cr.commit()

rows = []
for e in data["engines"]:
    eq = Equipment.search([("pdm_machine_ref", "=", e["ref"])])
    req = Request.search([("equipment_id", "=", eq.id), ("pdm_auto", "=", True)], order="pdm_cycle", limit=1)
    first = req.pdm_cycle if req else None
    expected = e["offline_first_alert"]
    expected = max(expected, e["failure_cycle"] - START) if expected is not None else None
    rows.append(
        {
            "ref": e["ref"],
            "request_cycle": first,
            "failure": e["failure_cycle"],
            "notice": e["failure_cycle"] - first if first else None,
            "same_as_offline": first == expected,
        }
    )
notices = [r["notice"] for r in rows if r["notice"] is not None]
print(
    json.dumps(
        {
            "engines": len(rows),
            "requests_before_failure": sum(r["notice"] is not None and r["notice"] > 0 for r in rows),
            f"with_at_least_{data['lead_time']}_cycles_notice": sum(n >= data["lead_time"] for n in notices),
            "median_notice_cycles": sorted(notices)[len(notices) // 2] if notices else None,
            "min_notice_cycles": min(notices) if notices else None,
            "request_cycle_same_as_offline": sum(r["same_as_offline"] for r in rows),
        }
    )
)
