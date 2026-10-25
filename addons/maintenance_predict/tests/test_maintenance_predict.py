from unittest.mock import MagicMock, patch

import requests

from odoo.exceptions import UserError
from odoo.tests import TransactionCase, tagged

GET = "odoo.addons.maintenance_predict.models.maintenance.requests.get"


def _response(payload):
    r = MagicMock()
    r.json.return_value = payload
    r.raise_for_status.return_value = None
    return r


def _pred(ref, rul, alert=False, failed=False, cycle=100):
    return {
        "ref": ref,
        "cycle": cycle,
        "failure_probability": 0.9 if alert else 0.05,
        "predicted_rul": rul,
        "health": min(rul / 125, 1) * 100,
        "alert": alert,
        "failed": failed,
    }


@tagged("post_install", "-at_install")
class TestMaintenancePredict(TransactionCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        Equipment = cls.env["maintenance.equipment"]
        cls.eq1 = Equipment.create({"name": "Engine 1", "pdm_machine_ref": "ENG-001"})
        cls.eq2 = Equipment.create({"name": "Engine 2", "pdm_machine_ref": "ENG-002"})
        cls.unlinked = Equipment.create({"name": "Forklift"})

    def _update(self, preds):
        with patch(GET, return_value=_response(preds)) as get:
            self.env["maintenance.equipment"].action_pdm_update_all()
        return get

    def test_only_linked_equipment_is_sent(self):
        get = self._update([_pred("ENG-001", 90), _pred("ENG-002", 40)])
        self.assertEqual(get.call_args.kwargs["params"]["refs"], "ENG-001,ENG-002")
        self.assertEqual(self.eq1.pdm_status, "ok")
        self.assertEqual(self.eq2.pdm_status, "watch")
        self.assertFalse(self.unlinked.pdm_status)

    def test_alert_creates_one_preventive_request(self):
        self._update([_pred("ENG-001", 18, alert=True, cycle=150), _pred("ENG-002", 90)])
        requests_ = self.eq1.pdm_request_ids
        self.assertEqual(len(requests_), 1)
        self.assertEqual(requests_.maintenance_type, "preventive")
        self.assertEqual(requests_.pdm_cycle, 150)
        self.assertEqual(self.eq1.pdm_status, "alert")
        self.assertFalse(self.eq2.pdm_request_ids)
        # A second alert while the request is open does not duplicate it
        self._update([_pred("ENG-001", 12, alert=True, cycle=155), _pred("ENG-002", 90)])
        self.assertEqual(len(self.eq1.pdm_request_ids), 1)
        # Once that request is done, a new alert opens a new one
        requests_.state = "done"
        self._update([_pred("ENG-001", 10, alert=True, cycle=160), _pred("ENG-002", 90)])
        self.assertEqual(len(self.eq1.pdm_request_ids), 2)

    def test_failed_machine(self):
        self._update([_pred("ENG-001", 0, alert=True, failed=True), _pred("ENG-002", 90)])
        self.assertEqual(self.eq1.pdm_status, "failed")
        self.assertFalse(self.eq1.pdm_request_ids, "No request is opened for a machine that has already failed")

    def test_service_down(self):
        with patch(GET, side_effect=requests.ConnectionError("down")):
            with self.assertRaises(UserError):
                self.env["maintenance.equipment"].action_pdm_update_all()
            self.env["maintenance.equipment"]._cron_pdm_update()  # logs and continues

    def test_regular_employee_can_update_predictions(self):
        """Runs as a real employee (not the superuser): maintenance users only read equipment."""
        from odoo.tests import new_test_user

        employee = new_test_user(self.env, login="tech", groups="base.group_user")
        with patch(GET, return_value=_response([_pred("ENG-001", 18, alert=True), _pred("ENG-002", 90)])):
            self.env["maintenance.equipment"].with_user(employee).action_pdm_update_all()
        self.assertEqual(self.eq1.pdm_status, "alert")
        self.assertEqual(len(self.eq1.pdm_request_ids), 1)
