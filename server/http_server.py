"""HTTP sidecar for the offload-harness: the same tools server.py exposes over
MCP, over plain JSON-over-HTTP on loopback, so a Go client can call the NPU.

    hailo-http.cmd            (or)  python server/http_server.py --listen 127.0.0.1:18813

Contract (consumed by the harness's internal/hailoclient):
    GET  /health            -> hailo_status() dict, always 200
    POST /v1/<tool>  {json} -> that tool's dict, 200 (structured error dicts are
                               results, not HTTP errors); 404 unknown tool; 400 bad body

The process EXITS after HAILO_SIDECAR_IDLE_SEC seconds (default 300) without a
request — the harness spawns it on demand, so nothing lingers when the editor is
not using AI features and there is no scheduler to clean up.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)  # server.py's own `from hailo_runtime import ...`


def _load_mcp_module():
    """Load server.py (the MCP module; its tool functions are plain callables)
    by path. A bare `import server` is ambiguous: when this file is imported
    as `server.http_server` (unittest run from the repo root does exactly that)
    the name already resolves to the server/ directory, not the module."""
    spec = importlib.util.spec_from_file_location("hailo_mcp_server", os.path.join(_HERE, "server.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


server = _load_mcp_module()

TOOLS = {
    "face_detect": server.hailo_face_detect,
    "face_embed": server.hailo_face_embed,
    "object_detect": server.hailo_object_detect,
    "person_embed": server.hailo_person_embed,
    "depth": server.hailo_depth,
    "enhance_low_light": server.hailo_enhance_low_light,
    "ocr": server.hailo_ocr,
    "embed": server.hailo_embed,
    "pose": server.hailo_pose,
    "segment": server.hailo_segment,
    "text_embed": server.hailo_text_embed,
    "zero_shot": server.hailo_zero_shot,
    "transcribe": server.hailo_transcribe,
}

_last_request = time.monotonic()
_lock = threading.Lock()  # one VDevice, one in-flight inference at a time


def _touch() -> None:
    global _last_request
    _last_request = time.monotonic()


class Handler(BaseHTTPRequestHandler):
    server_version = "hailo-sidecar/1.0"

    def log_message(self, fmt, *args):  # quiet by default; errors still surface
        if os.environ.get("HAILO_SIDECAR_LOG") == "1":
            super().log_message(fmt, *args)

    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        _touch()
        if self.path != "/health":
            return self._send(404, {"error": True, "kind": "unknown_path", "message": self.path})
        return self._send(200, server.hailo_status())

    def do_POST(self):
        _touch()
        if not self.path.startswith("/v1/"):
            return self._send(404, {"error": True, "kind": "unknown_path", "message": self.path})
        tool = TOOLS.get(self.path[len("/v1/"):])
        if tool is None:
            return self._send(404, {"error": True, "kind": "unknown_tool", "message": self.path})
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            args = json.loads(raw or b"{}")
        except json.JSONDecodeError as e:
            return self._send(400, {"error": True, "kind": "bad_request", "message": f"body is not JSON: {e}"})
        if not isinstance(args, dict):
            return self._send(400, {"error": True, "kind": "bad_request", "message": "body must be a JSON object of tool arguments"})
        try:
            with _lock:
                result = tool(**args)
        except TypeError as e:  # wrong/missing keyword → the caller's problem, said plainly
            return self._send(400, {"error": True, "kind": "bad_request", "message": str(e)})
        return self._send(200, result)


def make_server(host: str, port: int, idle_sec: int) -> ThreadingHTTPServer:
    """Build (not start) the server. idle_sec<=0 disables the idle exit (tests)."""
    srv = ThreadingHTTPServer((host, port), Handler)
    if idle_sec > 0:
        def reaper():
            while True:
                time.sleep(5)
                if time.monotonic() - _last_request > idle_sec:
                    srv.shutdown()
                    return
        threading.Thread(target=reaper, daemon=True).start()
    return srv


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--listen", default="127.0.0.1:18813", help="host:port (loopback only)")
    ap.add_argument("--idle-sec", type=int, default=int(os.environ.get("HAILO_SIDECAR_IDLE_SEC", "300")))
    a = ap.parse_args()
    host, _, port = a.listen.rpartition(":")
    if host not in ("127.0.0.1", "localhost"):
        print("refusing to bind a non-loopback address; the sidecar is not an authenticated service", file=sys.stderr)
        return 2
    srv = make_server(host, int(port), a.idle_sec)
    print(f"hailo sidecar listening on {a.listen} (idle exit {a.idle_sec}s)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
