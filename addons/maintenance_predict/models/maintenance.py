import logging
from datetime import timedelta

import requests

from odoo import api, fields, models
from odoo.exceptions import UserError

_logger = logging.getLogger(__name__)

SERVICE_TIMEOUT = 30
STATUS = [("ok", "Healthy"), ("watch", "Watch"), ("alert", "Maintenance Needed"), ("failed", "Failed")]
WATCH_RUL = 50  # cycles: show "Watch" before the service's maintenance alert


class MaintenanceEquipment(models.Model):
    _inherit = "maintenance.equipment"

    pdm_machine_ref = fields.Char(
        string="Sensor Machine ID",
        copy=False,
        index=True,
        help="Identifier of this equipment in the prediction service (e.g. ENG-069).",
    )
    pdm_health = fields.Float(string="Health (%)", readonly=True, copy=False, digits=(5, 1), aggregator="avg")
    pdm_predicted_rul = fields.Float(
        string="Predicted Remaining Life (cycles)", readonly=True, copy=False, digits=(6, 1), aggregator="min"
    )
    pdm_failure_probability = fields.Float(
        string="Failure Risk, Next 30 Cycles (%)", readonly=True, copy=False, digits=(5, 1)
    )
    pdm_cycle = fields.Integer(string="Current Cycle", readonly=True, copy=False)
    pdm_status = fields.Selection(STATUS, string="Predicted Status", readonly=True, copy=False, index=True)
    pdm_scored_on = fields.Datetime(string="Last Prediction", readonly=True, copy=False)
    pdm_request_ids = fields.One2many(
        "maintenance.request", "equipment_id", domain=[("pdm_auto", "=", True)], string="Predicted Requests"
    )

    _pdm_machine_ref_uniq = models.Constraint("unique(pdm_machine_ref)", "Each sensor machine ID can be linked once.")

    @api.model
    def _pdm_service_url(self):
        url = self.env["ir.config_parameter"].sudo().get_str("maintenance_predict.service_url")
        return (url or "http://ml:8000").rstrip("/")

    def _pdm_fetch(self):
        url = self._pdm_service_url()
        refs = ",".join(self.mapped("pdm_machine_ref"))
        try:
            response = requests.get(f"{url}/predict", params={"refs": refs}, timeout=SERVICE_TIMEOUT)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            _logger.warning("Prediction service error: %s", exc)
            raise UserError(self.env._("The prediction service at %(url)s is not available.", url=url)) from exc

    def _pdm_update(self):
        """Store the latest prediction; open one preventive request when the service raises an alert."""
        equipment = self.filtered("pdm_machine_ref")
        if not equipment:
            return 0
        by_ref = {p["ref"]: p for p in equipment._pdm_fetch()}
        now = fields.Datetime.now()
        created = 0
        for eq in equipment:
            p = by_ref.get(eq.pdm_machine_ref)
            if not p:
                continue
            if p["failed"]:
                status = "failed"
            elif p["alert"]:
                status = "alert"
            else:
                status = "watch" if p["predicted_rul"] <= WATCH_RUL else "ok"
            # Prediction fields and automatic requests come from the service, not from the user, so they are
            # written with elevated rights (as the scheduled job does). Users still only update equipment they see.
            eq.sudo().write(
                {
                    "pdm_health": p["health"],
                    "pdm_predicted_rul": p["predicted_rul"],
                    "pdm_cycle": p["cycle"],
                    "pdm_failure_probability": p["failure_probability"] * 100,
                    "pdm_status": status,
                    "pdm_scored_on": now,
                }
            )
            if status == "alert" and not eq.sudo()._pdm_open_request():
                eq.sudo()._pdm_create_request(p)
                created += 1
        return created

    def _pdm_open_request(self):
        self.ensure_one()
        return self.env["maintenance.request"].search(
            [
                ("equipment_id", "=", self.id),
                ("pdm_auto", "=", True),
                ("state", "not in", ("done", "cancelled")),
            ],
            limit=1,
        )

    def _pdm_create_request(self, p):
        self.ensure_one()
        return self.env["maintenance.request"].create(
            {
                "name": self.env._("Predicted failure: %(name)s", name=self.name),
                "equipment_id": self.id,
                "maintenance_type": "preventive",
                "priority": "3",
                "schedule_date": fields.Datetime.now() + timedelta(days=1),
                "pdm_auto": True,
                "pdm_predicted_rul": p["predicted_rul"],
                "pdm_cycle": p["cycle"],
                "description": self.env._(
                    "Created automatically at cycle %(cycle)s: predicted remaining life %(rul).0f cycles, "
                    "failure risk in the next 30 cycles %(risk).0f%%.",
                    cycle=p["cycle"],
                    rul=p["predicted_rul"],
                    risk=p["failure_probability"] * 100,
                ),
            }
        )

    @api.model
    def action_pdm_update_all(self):
        created = self.search([("pdm_machine_ref", "!=", False)])._pdm_update()
        return {
            "type": "ir.actions.client",
            "tag": "display_notification",
            "params": {
                "title": self.env._("Predictions updated"),
                "message": self.env._("%(count)s new maintenance request(s).", count=created),
                "type": "success",
                "next": {"type": "ir.actions.act_window_close"},
            },
        }

    def action_pdm_update(self):
        self._pdm_update()
        return True

    @api.model
    def _cron_pdm_update(self):
        try:
            self.search([("pdm_machine_ref", "!=", False)])._pdm_update()
        except UserError as exc:
            _logger.warning("Scheduled prediction update skipped: %s", exc)


class MaintenanceRequest(models.Model):
    _inherit = "maintenance.request"

    pdm_auto = fields.Boolean(string="Created by Prediction", readonly=True, copy=False, index=True)
    pdm_predicted_rul = fields.Float(string="Predicted Remaining Life at Creation", readonly=True, copy=False)
    pdm_cycle = fields.Integer(string="Cycle at Creation", readonly=True, copy=False)
