"""Local HTTP demo. Checkpoint-free mode exposes the labelled colour rule only."""

import base64
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import mimetypes
from pathlib import Path
import threading
from urllib.parse import parse_qs, urlsplit

import numpy as np
from PIL import Image

from .data import (BACKGROUND_NAMES, RENDERER_VERSION, SHAPES, foreground_hash,
                   make_objects, render)
from .metrics import colour_only_control

WEB_ROOT = Path(__file__).parent / "web"


class Demo:
    def __init__(self, runs=None):
        self.predictors = {}
        self.lock = threading.Lock()
        if runs:
            root = Path(runs)
            if not root.is_dir():
                raise FileNotFoundError("Checkpoint directory does not exist")
            files = sorted(root.rglob("final.pt"))
            if not files:
                raise ValueError("No final.pt checkpoints in the requested directory")
            from .inference import Predictor
            for index, path in enumerate(files):
                self.predictors[f"model-{index}"] = Predictor.load(path)

    def catalog(self):
        models = []
        for key, predictor in self.predictors.items():
            m = predictor.metadata
            models.append({"id": key, **{field: m[field] for field in ("architecture", "condition", "seed", "parameters", "checkpoint_sha256")},
                           "protocol": m["config"]["protocol"]})
        return {"shapes": SHAPES, "backgrounds": BACKGROUND_NAMES, "renderer": RENDERER_VERSION,
                "models": models, "mode": "checkpoints" if models else "colour-rule-only",
                "colour_only_metrics": colour_only_control()}

    def example(self, shape, seed):
        if shape not in range(3) or not 0 <= seed <= 2**32 - 1:
            raise ValueError("Shape must be 0–2 and seed must be an unsigned 32-bit integer")
        obj = make_objects(3, "demo", seed)[shape]
        images = []
        for background in range(3):
            image = Image.fromarray(render(obj, background)[0])
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            images.append("data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode())
        predictions = {"colour-rule": np.eye(3).tolist()}
        with self.lock:
            for key, predictor in self.predictors.items():
                predictions[key] = predictor.objects([obj])[0].tolist()
        return {"object": asdict(obj), "foreground_hash": foreground_hash(obj), "images": images, "predictions": predictions}


def make_handler(demo):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, status, data, content_type="application/json"):
            body = json.dumps(data, allow_nan=False).encode() if content_type == "application/json" else data
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            url = urlsplit(self.path)
            try:
                if url.path == "/api/health":
                    return self.send(200, {"status": "ok", "mode": demo.catalog()["mode"]})
                if url.path == "/api/catalog":
                    return self.send(200, demo.catalog())
                if url.path == "/api/example":
                    query = parse_qs(url.query, max_num_fields=8)
                    shape = int(query.get("shape", ["0"])[0])
                    seed = int(query.get("seed", ["73"])[0])
                    return self.send(200, demo.example(shape, seed))
                assets = {"/": "index.html", "/index.html": "index.html", "/style.css": "style.css", "/app.js": "app.js"}
                if url.path in assets:
                    path = WEB_ROOT / assets[url.path]
                    return self.send(200, path.read_bytes(), (mimetypes.guess_type(path.name)[0] or "application/octet-stream") + "; charset=utf-8")
                return self.send(404, {"error": "Not found"})
            except (ValueError, OverflowError):
                return self.send(400, {"error": "Invalid shape or seed"})
            except Exception:
                return self.send(500, {"error": "Prediction failed; inspect the checkpoint and server environment"})

    return Handler


def serve(runs=None, host="127.0.0.1", port=8000):
    demo = Demo(runs)
    server = ThreadingHTTPServer((host, port), make_handler(demo))
    print(f"Shortcut Sleuth: http://{host}:{server.server_port} ({demo.catalog()['mode']})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
