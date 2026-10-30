import base64
from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer
import io
import json
from types import SimpleNamespace
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np
from PIL import Image

from shortcut_sleuth.data import foreground_hash, make_objects, render
from shortcut_sleuth.server import Demo, make_handler


class ServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.handler = make_handler(Demo())

    def request(self, route, disconnect=False):
        class Connection:
            def __init__(self):
                self.body = bytearray()

            def makefile(self, *_):
                return io.BytesIO(f"GET {route} HTTP/1.0\r\nHost: localhost\r\n\r\n".encode())

            def sendall(self, data):
                if disconnect:
                    raise BrokenPipeError("Client disconnected")
                self.body.extend(data)

        connection = Connection()
        self.handler(connection, ("127.0.0.1", 0), SimpleNamespace())
        if disconnect:
            return
        head, body = bytes(connection.body).split(b"\r\n\r\n", 1)
        lines = head.decode().split("\r\n")
        return int(lines[0].split()[1]), dict(line.split(": ", 1) for line in lines[1:]), body

    def get(self, route):
        status, _, body = self.request(route)
        if status != 200:
            raise HTTPError(route, status, body.decode(), {}, None)
        return json.loads(body)

    def test_health_catalog_and_real_assets(self):
        self.assertEqual(self.get("/api/health")["status"], "ok")
        catalog = self.get("/api/catalog")
        self.assertEqual(catalog["models"], [])
        self.assertEqual(catalog["mode"], "colour-rule-only")
        for route, marker in (("/", b"Shortcut Sleuth"), ("/app.js", b"loadExample"), ("/style.css", b"--accent")):
            status, headers, body = self.request(route)
            self.assertEqual(status, 200)
            self.assertIn(marker, body)
            self.assertIn("nosniff", headers["X-Content-Type-Options"])

    def test_http_exact_render_and_control_predictions(self):
        example = self.get("/api/example?shape=2&seed=19")
        obj = make_objects(3, "demo", 19)[2]
        self.assertEqual(example["foreground_hash"], foreground_hash(obj))
        for b, uri in enumerate(example["images"]):
            image = np.array(Image.open(io.BytesIO(base64.b64decode(uri.split(",")[1]))))
            np.testing.assert_array_equal(image, render(obj, b)[0])
        np.testing.assert_array_equal(example["predictions"]["colour-rule"], np.eye(3))
        self.assertEqual(example, self.get("/api/example?shape=2&seed=19"))

    def test_bad_requests_and_path_allowlist(self):
        for route, status in (("/api/example?shape=9", 400), ("/api/example?seed=-1", 400),
                              ("/api/example?seed=word", 400), ("/api/example?seed=4294967296", 400),
                              ("/../README.md", 404), ("/api/unknown", 404)):
            with self.subTest(route=route), self.assertRaises(HTTPError) as error:
                self.get(route)
            self.assertEqual(error.exception.code, status)
            error.exception.close()

    def test_concurrent_requests(self):
        with ThreadPoolExecutor(max_workers=4) as pool:
            examples = list(pool.map(lambda seed: self.get(f"/api/example?seed={seed}"), range(8)))
        self.assertEqual(len({item["foreground_hash"] for item in examples}), 8)

    def test_disconnected_client(self):
        self.request("/api/catalog", disconnect=True)

    def test_socket_http_roundtrip(self):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", 0), self.handler)
        except PermissionError:
            self.skipTest("Environment prohibits local sockets; handlers are checked in memory")
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with urlopen(f"http://127.0.0.1:{server.server_port}/api/health", timeout=5) as response:
                self.assertEqual(json.load(response)["status"], "ok")
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
