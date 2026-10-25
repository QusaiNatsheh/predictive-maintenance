# LinkedIn posts: Predictive Maintenance for Odoo 20 (drafts)

Drafts for review before posting. Replace `<repo link>` with the GitHub URL once the repository is public.
Every number comes from the README.

---

## Post 1: from sensor data to a maintenance request

**Attach:** a simple architecture diagram (sensors → FastAPI + LightGBM → Odoo Maintenance), or the README "How it
works" section

A failure prediction is only useful if someone acts on it. So I connected one to Odoo.

How it works:
1. **Sensors:** a replay sends engine sensor readings, one cycle at a time, from engines the model has never seen.
2. **Model:** a LightGBM model behind a FastAPI service estimates each machine's remaining life and its risk of
   failing in the next 30 cycles.
3. **Decision:** a cost-based rule decides when to act: plan maintenance when the predicted life is 21 cycles or
   fewer.
4. **Odoo 20:** the equipment shows its health, and a preventive maintenance request opens automatically. Only one
   request per machine, and none for a machine that has already failed.

If the service is down, Odoo logs a warning and keeps working. MLflow tracks every model version.

<repo link>

#Odoo #PredictiveMaintenance #MachineLearning #Manufacturing #Python

---

## Post 2: is the model worth it? The cost comparison

**Attach:** `docs/cost_curve.png`

"Predictive maintenance saves money" is easy to say. I measured it on NASA's turbofan engine data, on 20 engines
the model never saw during training or tuning.

Cost per engine (a planned maintenance costs 1, an unplanned failure costs 10):
• Run to failure: 10.00 (20 of 20 engines fail)
• Fixed schedule, every 118 cycles: 1.37
• A simple sensor-drift threshold: 1.16
• The model (act when predicted life ≤ 21 cycles): 1.10, with 0 missed failures

So the model costs 89% less than running to failure, but only 5% less than a simple sensor rule.

The honest lesson: most of the value comes from monitoring at all. The model's own advantage is smaller: it acts
later without missing failures, so less useful life is thrown away.

Remaining-life error: RMSE 13.1 cycles, against 36.7 for a fleet-average baseline.

<repo link>

#PredictiveMaintenance #MachineLearning #DataScience #Maintenance

---

## Post 3: watching it work in Odoo

**Attach:** `docs/screenshots/demo.gif` (or a short screen recording)

What predictive maintenance looks like inside Odoo 20:

1. **Fleet health:** every machine is Healthy, Watch or Maintenance Needed, sorted by predicted remaining life.
2. **10 cycles later:** more machines cross the line, and Odoo opens a preventive request for each one.
3. **Equipment:** a Predictive Health tab with the current cycle, health %, predicted life and 30-cycle risk.
4. **Requests:** high priority, marked "Created by Prediction", with the cycle and predicted life at creation.

In the end-to-end test, all 20 engines got their maintenance request before failing, with at least 15 cycles of
notice (median 20).

Built with Odoo 20 Community, FastAPI, LightGBM and MLflow. One `make demo` runs it.

<repo link>

#Odoo #Maintenance #Manufacturing #Python
