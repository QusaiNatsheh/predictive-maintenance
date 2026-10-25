import requests

from odoo import fields, models
from odoo.exceptions import UserError


class ResConfigSettings(models.TransientModel):
    _inherit = "res.config.settings"

    pdm_service_url = fields.Char(
        string="Prediction Service URL", config_parameter="maintenance_predict.service_url", default="http://ml:8000"
    )

    def _pdm_post(self, path, payload):
        url = self.env["maintenance.equipment"]._pdm_service_url()
        try:
            response = requests.post(f"{url}{path}", json=payload, timeout=30)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise UserError(self.env._("The prediction service at %(url)s is not available.", url=url)) from exc

    def action_pdm_update_now(self):
        self.execute()
        return self.env["maintenance.equipment"].action_pdm_update_all()

    def action_pdm_simulator_tick(self):
        """Demo only: advance the service's sensor replay by 5 cycles, then refresh predictions."""
        self._pdm_post("/simulator/tick", {"cycles": 5})
        return self.env["maintenance.equipment"].action_pdm_update_all()

    def action_pdm_simulator_reset(self):
        """Demo only: restart the replay with machines spread between 15 and 124 cycles before failure."""
        self._pdm_post("/simulator/reset", {"staggered": True})
        return self.env["maintenance.equipment"].action_pdm_update_all()
