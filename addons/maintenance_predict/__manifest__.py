{
    "name": "Predictive Maintenance",
    "summary": "Equipment health from sensor-based failure prediction, with automatic maintenance requests",
    "version": "20.0.1.0.0",
    "category": "Manufacturing/Maintenance",
    "author": "Qusai Natsheh",
    "website": "https://github.com/QusaiNatsheh/predictive-maintenance",
    "license": "LGPL-3",
    "depends": ["maintenance"],
    "external_dependencies": {"python": ["requests"]},
    "data": [
        "data/ir_cron.xml",
        "views/maintenance_views.xml",
        "views/res_config_settings_views.xml",
    ],
}
