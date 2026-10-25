# Local demo with Docker. `make demo` builds (downloads + verifies the datasets, trains), creates database "demo",
# links 20 equipment to the replayed engines, and replays them to failure through Odoo.
DB ?= demo

up:
	docker compose up -d --build

demo: up
	docker compose exec odoo odoo -d $(DB) -i maintenance_predict --without-demo=all --stop-after-init
	docker compose exec -T -e DEMO_JSON=/mnt/data/demo.json odoo odoo shell -d $(DB) --no-http < scripts/load_demo_data.py
	docker compose exec -T -e DEMO_JSON=/mnt/data/demo.json odoo odoo shell -d $(DB) --no-http < scripts/e2e_check.py
	@echo "Open http://localhost:8072 (database $(DB), user admin / admin) -> Maintenance -> Predictive Health"

e2e:
	docker compose exec -T -e DEMO_JSON=/mnt/data/demo.json odoo odoo shell -d $(DB) --no-http < scripts/e2e_check.py

reset-demo:  # fresh demo database with the 20 engines linked (run `make e2e` to replay them to failure)
	docker compose exec -T -e PGPASSWORD=odoo odoo dropdb --force --if-exists -h db -U odoo $(DB)
	docker compose exec odoo odoo -d $(DB) -i maintenance_predict --without-demo=all --stop-after-init
	docker compose exec -T -e DEMO_JSON=/mnt/data/demo.json odoo odoo shell -d $(DB) --no-http < scripts/load_demo_data.py
	docker compose restart odoo

screenshots:  # needs: pip install playwright pillow (uses the installed Chrome)
	python scripts/screenshots.py --url http://localhost:8072 --service http://localhost:8003 --db $(DB)

down:
	docker compose down

.PHONY: up demo e2e reset-demo screenshots down
