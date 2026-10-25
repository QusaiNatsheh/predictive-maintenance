"""Create one Odoo equipment per replayed engine. Run inside `odoo shell`:
DEMO_JSON=/path/demo.json odoo shell -d <db> < scripts/load_demo_data.py"""

import json
import os

data = json.load(open(os.environ.get("DEMO_JSON", "/mnt/data/demo.json"), encoding="utf-8"))
Equipment = env["maintenance.equipment"]
category = env["maintenance.equipment.category"].search([("name", "=", "Turbofan Engines")], limit=1) or env[
    "maintenance.equipment.category"
].create({"name": "Turbofan Engines"})
created = 0
for e in data["engines"]:
    if not Equipment.search_count([("pdm_machine_ref", "=", e["ref"])]):
        Equipment.create(
            {
                "name": f"Turbofan engine {e['ref']}",
                "pdm_machine_ref": e["ref"],
                "category_id": category.id,
                "serial_no": e["ref"],
            }
        )
        created += 1
env.cr.commit()
print(f"{created} equipment created ({len(data['engines'])} engines in the demo)")
