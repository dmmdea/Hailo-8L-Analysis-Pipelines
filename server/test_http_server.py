"""HTTP sidecar contract tests. Run with HAILO_VISION_ENABLED=0 so no device is
touched: every tool then returns its structured disabled dict, which is exactly
the routing/JSON behaviour this layer owns."""
import json
import os
import sys
import threading
import unittest
import urllib.error
import urllib.request

os.environ["HAILO_VISION_ENABLED"] = "0"
# server/ for http_server + server.py; shared/ for vision_shared, which
# hailo_runtime imports — so the file runs with no ambient PYTHONPATH.
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shared"))
import http_server  # noqa: E402


class SidecarContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = http_server.make_server("127.0.0.1", 0, idle_sec=0)
        cls.port = cls.srv.server_address[1]
        cls.t = threading.Thread(target=cls.srv.serve_forever, daemon=True)
        cls.t.start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def _get(self, path):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}") as r:
            return r.status, json.loads(r.read())

    def _post(self, path, body):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", data=data,
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_health_reports_status_dict(self):
        status, body = self._get("/health")
        self.assertEqual(status, 200)
        self.assertIn("enabled", body)
        self.assertFalse(body["enabled"])

    def test_tool_routes_to_structured_result(self):
        status, body = self._post("/v1/face_detect", {"image_path": "nope.jpg"})
        self.assertEqual(status, 200)
        self.assertTrue(body.get("error"))
        self.assertEqual(body.get("kind"), "hailo_disabled")

    def test_unknown_tool_is_404(self):
        status, body = self._post("/v1/teleport", {})
        self.assertEqual(status, 404)
        self.assertEqual(body.get("kind"), "unknown_tool")

    def test_bad_json_is_400(self):
        status, body = self._post("/v1/face_detect", b"{not json")
        self.assertEqual(status, 400)
        self.assertEqual(body.get("kind"), "bad_request")

    def test_non_object_body_is_400(self):
        status, body = self._post("/v1/face_detect", [1, 2, 3])
        self.assertEqual(status, 400)

    def test_every_mcp_tool_is_routable(self):
        for name in ("face_detect", "face_embed", "object_detect", "person_embed",
                     "depth", "enhance_low_light", "ocr", "embed"):
            status, _ = self._post(f"/v1/{name}", {"image_path": "nope.jpg"})
            self.assertEqual(status, 200, name)


if __name__ == "__main__":
    unittest.main()
