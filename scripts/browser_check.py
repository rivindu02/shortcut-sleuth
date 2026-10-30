"""Exercise the actual local HTTP application in Chromium and save evidence."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import threading
from http.server import ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", help="Checkpoint root; omit to check the renderer/control mode")
    parser.add_argument("--output", default="artifacts/browser")
    args = parser.parse_args()
    from playwright.sync_api import expect, sync_playwright
    from shortcut_sleuth.server import Demo, make_handler
    from shortcut_sleuth.data import save_json

    if args.runs:
        import torch
        torch.set_num_threads(2)
    demo = Demo(args.runs)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(demo))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    errors, checks, checked_models = [], [], set()
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True, args=["--disable-gpu"])
            try:
                page = browser.new_page(viewport={"width": 1440, "height": 1100}, device_scale_factor=1)
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(f"http://127.0.0.1:{server.server_port}/", wait_until="networkidle")

                def wait_for_object():
                    expect(page.locator("#specimen-image")).to_be_visible()
                    expect(page.locator("#specimen-image")).to_have_js_property("naturalWidth", 64)
                    expect(page.locator("#object-id")).to_contain_text("mask")

                wait_for_object()
                checks.append("page_load_and_actual_image")
                original_object = page.locator("#object-id").inner_text()
                original_image = page.locator("#specimen-image").get_attribute("src")
                page.locator('[data-background="1"]').click()
                assert page.locator("#object-id").inner_text() == original_object
                assert page.locator("#specimen-image").get_attribute("src") != original_image
                assert "Square" in page.locator("#baseline-prediction").inner_text()
                checks.append("background_swap_preserves_object")
                page.screenshot(path=str(output / "lab-intervention.png"), full_page=True)
                page.locator('[data-shape="1"]').click()
                wait_for_object()
                assert "Square" in page.locator("#specimen-image").get_attribute("alt")
                previous = page.locator("#object-id").inner_text()
                page.locator("#new-object").click()
                wait_for_object()
                assert page.locator("#object-id").inner_text() != previous
                assert page.locator("#object-seed").input_value() == "74"
                checks.append("shape_and_seed_controls")

                if demo.predictors:
                    for condition in ("correlated", "independent"):
                        page.locator("#condition").select_option(condition)
                        cards = page.locator(".model-card")
                        assert cards.count() == 2
                        expected = page.request.get(f"http://127.0.0.1:{server.server_port}/api/example?shape=1&seed=74").json()
                        for index in range(2):
                            card = cards.nth(index)
                            choice = card.locator(".model-choice")
                            for option in range(choice.locator("option").count()):
                                choice.select_option(index=option)
                                model_id = choice.input_value()
                                for background in range(3):
                                    page.locator(f'[data-background="{background}"]').click()
                                    shown = card.locator(".probability-row span:last-child").all_text_contents()
                                    actual = expected["predictions"][model_id][background]
                                    assert shown == [f"{100 * probability:.1f}%" for probability in actual], (shown, actual)
                                checked_models.add(model_id)
                    assert checked_models == set(demo.predictors)
                    checks.append("all_conditions_seeds_and_backgrounds_match_actual_API_probabilities")
                    page.locator("#condition").select_option("correlated")
                    for choice in page.locator(".model-choice").all():
                        choice.select_option(index=0)
                else:
                    assert page.locator("#condition").is_disabled()
                    assert page.get_by_text("Awaiting a checkpoint", exact=True).count() == 2
                    checks.append("checkpoint_free_empty_state")

                # Capture the actual page after returning to a known object.
                page.locator("#object-seed").fill("73")
                page.locator("#object-seed").dispatch_event("change")
                wait_for_object()
                page.locator('[data-shape="0"]').click()
                wait_for_object()
                page.locator('[data-background="0"]').click()
                assert not page.locator("#error").is_visible()
                page.screenshot(path=str(output / "lab-desktop.png"), full_page=True)
                page.set_viewport_size({"width": 390, "height": 844})
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth + 1")
                page.screenshot(path=str(output / "lab-mobile.png"), full_page=True)
                checks.append("desktop_and_mobile_layout")
                assert not errors, errors
                save_json(output / "browser.json", {"passed": True, "checks": checks,
                          "javascript_errors": errors, "browser": browser.version,
                          "playwright": importlib.metadata.version("playwright"), "catalog": demo.catalog(),
                          "checkpoint_ids_checked": sorted(checked_models),
                          "capture": {"object_id": "demo-73-000000", "seed": 73,
                                      "canonical_background": 0, "intervention_background": 1}})
                print(json.dumps({"passed": True, "checks": checks}, indent=2))
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


if __name__ == "__main__":
    main()
