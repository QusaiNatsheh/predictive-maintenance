"""Take README screenshots and a demo GIF from the running Docker demo (`make reset-demo` first).

The replay is restarted "staggered" so the fleet shows healthy, watched and alerting machines together, then
advanced 10 cycles so new maintenance requests appear.
Needs Playwright with a Chrome/Chromium install:  pip install playwright pillow
Usage: python scripts/screenshots.py [--url http://localhost:8072] [--service http://localhost:8003]
"""

import argparse
import io
import json
import urllib.request
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

VIEWPORT = {"width": 1440, "height": 900}
HIDE_TOURS = (
    "document.addEventListener('DOMContentLoaded', () => { const s = document.createElement('style');"
    " s.textContent = '.o_tour_pointer { display: none !important; }'; document.head.appendChild(s); });"
)
RPC = """async ([model, method, args, kwargs]) => {
    const r = await fetch('/web/dataset/call_kw', {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({jsonrpc: '2.0', method: 'call', params: {model, method, args, kwargs}})});
    return (await r.json()).result;
}"""


def post(url: str, payload: dict) -> None:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=30).read()


def settle(page) -> None:
    """Odoo keeps a live connection open (bus), so wait for the rendered view instead of network idle."""
    page.wait_for_selector(".o_action_manager .o_view_controller", timeout=30000)
    page.wait_for_timeout(1500)
    page.mouse.move(5, 890)
    page.evaluate("document.activeElement && document.activeElement.blur()")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8072")
    parser.add_argument("--service", default="http://localhost:8003")
    parser.add_argument("--db", default="demo")
    parser.add_argument("--out", type=Path, default=Path("docs/screenshots"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    frames = []

    def shot(page, name: str, frame: bool = True) -> None:
        png = page.screenshot()
        (args.out / f"{name}.png").write_bytes(png)
        if frame:
            frames.append(Image.open(io.BytesIO(png)).convert("RGB"))

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport=VIEWPORT, device_scale_factor=1)
        page.add_init_script(HIDE_TOURS)
        page.goto(f"{args.url}/web/login?db={args.db}")
        page.fill("input[name=login]", "admin")
        page.fill("input[name=password]", "admin")
        page.click("button[type=submit]")
        page.wait_for_selector(".o_main_navbar", timeout=30000)

        # Clean start: no predicted requests, fleet spread over its life
        old = page.evaluate(RPC, ["maintenance.request", "search", [[["pdm_auto", "=", True]]], {}])
        if old:
            page.evaluate(RPC, ["maintenance.request", "unlink", [old], {}])
        post(f"{args.service}/simulator/reset", {"staggered": True})
        page.evaluate(RPC, ["maintenance.equipment", "action_pdm_update_all", [], {}])

        page.goto(f"{args.url}/odoo/action-maintenance_predict.action_equipment_health")
        settle(page)
        shot(page, "01_fleet_health")

        post(f"{args.service}/simulator/tick", {"cycles": 10})
        page.evaluate(RPC, ["maintenance.equipment", "action_pdm_update_all", [], {}])
        page.reload()
        settle(page)
        shot(page, "02_fleet_after_10_cycles")

        ids = page.evaluate(
            RPC,
            [
                "maintenance.equipment",
                "search",
                [[["pdm_status", "=", "alert"]]],
                {"limit": 1, "order": "pdm_predicted_rul asc"},
            ],
        )
        page.goto(f"{args.url}/odoo/action-maintenance_predict.action_equipment_health/{ids[0]}")
        page.wait_for_selector(".o_form_view")
        page.locator(".nav-link:has-text('Predictive Health')").first.click()
        page.wait_for_timeout(1200)
        page.mouse.move(5, 890)
        shot(page, "03_equipment_health_tab")

        page.goto(f"{args.url}/odoo/action-maintenance.hr_equipment_request_action")
        settle(page)
        page.locator(".o_searchview_input").fill("Predicted failure")
        page.keyboard.press("Enter")
        settle(page)
        shot(page, "04_predicted_requests")
        browser.close()

    gif = args.out / "demo.gif"
    small = [f.resize((960, 600), Image.Resampling.LANCZOS) for f in frames]
    small[0].save(gif, save_all=True, append_images=small[1:], duration=[3000, 3000, 4000, 3500], loop=0, optimize=True)
    print(f"{len(frames)} frames -> {gif}; screenshots in {args.out}")


if __name__ == "__main__":
    main()
