======================
Predictive Maintenance
======================

Shows a predicted health, remaining life and failure risk on maintenance equipment, and opens a preventive
maintenance request automatically when the prediction service says maintenance is needed.

Usage
=====

#. Set the service address in *Maintenance → Configuration → Settings → Predictive Maintenance*.
#. On each equipment, tab *Predictive Health*, enter its *Sensor Machine ID* in the prediction service.
#. Predictions refresh every 15 minutes (scheduled action) or with *Update All* in
   *Maintenance → Predictive Health*.
#. When the service raises an alert, one preventive request is created (priority high, marked *Created by
   Prediction*). No second request is opened while that one is still open.

The settings page also has demo buttons that advance or restart the service's sensor replay.

Credits
=======

Author: Qusai Natsheh. License: LGPL-3.
