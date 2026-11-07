"""M1.5 regression tests: DNS-rebinding defence, catch-all errors, llama.cpp API key.

Audit findings G1 and G2 (docs/audit-v0.10.md) were reproduced against v0.10 before
these fixes: a foreign Host header was served, and an internal exception dropped the
connection without a response.
"""
import contextlib
import http.client
import io
import json
import os
import socket
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock
from urllib.error import HTTPError

from cpu_first.core import LLAMA_API_KEY_ENV, LocalLlama, ModelRejectedRequest
from cpu_platform import launch
from cpu_platform.engine import LazyLocalModel
from cpu_platform.provenance import server_props
from cpu_platform.server import create_server
from tests.fake_llama import FakeLlama

LOGIC = json.dumps({"premises": ["p"], "conclusion": "p"})


class ServerCase(unittest.TestCase):
    server_kwargs: dict = {}

    def setUp(self):
        self.server = create_server(0, **self.server_kwargs)
        self.port = self.server.server_port
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": .05},
                                       daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def request(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        hdrs = {"Content-Type": "application/json"} if body is not None else {}
        hdrs.update(headers or {})
        conn.request(method, path, body, hdrs)
        response = conn.getresponse()
        raw = response.read()
        conn.close()
        return response.status, (json.loads(raw) if raw else None)

    def raw(self, request_bytes: bytes) -> bytes:
        with socket.create_connection(("127.0.0.1", self.port), timeout=5) as sock:
            sock.sendall(request_bytes)
            chunks = []
            while True:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                chunks.append(chunk)
        return b"".join(chunks)


class HostOriginGuard(ServerCase):
    def test_foreign_host_rejected_on_every_method(self):
        evil = {"Host": f"attacker.example:{self.port}"}
        self.assertEqual(self.request("GET", "/healthz", headers=evil), (403, {"error": "forbidden_host"}))
        self.assertEqual(self.request("POST", "/v1/verify-logic", LOGIC, evil)[0], 403)
        self.assertEqual(self.request("DELETE", "/v1/jobs/" + "0" * 32, headers=evil)[0], 403)
        self.assertEqual(self.request("GET", "/", headers=evil)[0], 403)

    def test_rebinding_lookalike_hosts_rejected(self):
        for host in ("127.0.0.1.attacker.example", "localhost.attacker.example",
                     "127.0.0.1@attacker.example", "attacker.example#127.0.0.1",
                     "0.0.0.0", "", "[::1", "127.0.0.2"):
            with self.subTest(host=host):
                self.assertEqual(self.request("GET", "/healthz", headers={"Host": host})[0], 403)

    def test_loopback_hosts_accepted(self):
        for host in (f"127.0.0.1:{self.port}", f"localhost:{self.port}", f"[::1]:{self.port}",
                     "LOCALHOST", "127.0.0.1"):
            with self.subTest(host=host):
                status, body = self.request("POST", "/v1/verify-logic", LOGIC, {"Host": host})
                self.assertEqual(status, 200)
                self.assertEqual(body["answer"], "entailed")

    def test_cross_site_and_null_origin_rejected(self):
        for origin in ("http://attacker.example", f"http://attacker.example:{self.port}", "null",
                       "file://", "ftp://127.0.0.1", "http://127.0.0.1.attacker.example"):
            with self.subTest(origin=origin):
                status, body = self.request("POST", "/v1/verify-logic", LOGIC, {"Origin": origin})
                self.assertEqual((status, body), (403, {"error": "forbidden_origin"}))

    def test_same_origin_playground_accepted(self):
        status, _ = self.request("POST", "/v1/verify-logic", LOGIC,
                                 {"Origin": f"http://127.0.0.1:{self.port}"})
        self.assertEqual(status, 200)

    def test_duplicate_host_header_rejected(self):
        reply = self.raw(b"GET /healthz HTTP/1.1\r\nHost: 127.0.0.1\r\nHost: attacker.example\r\n"
                         b"Connection: close\r\n\r\n")
        self.assertTrue(reply.startswith(b"HTTP/1.1 403"), reply[:40])

    def test_missing_host_is_allowed_for_http10_tools(self):
        # Browsers always send Host, so an absent header is not a rebinding vector.
        reply = self.raw(b"GET /healthz HTTP/1.0\r\n\r\n")
        self.assertTrue(reply.startswith(b"HTTP/1.1 200"), reply[:40])

    def test_guard_runs_before_auth_and_does_not_leak_auth_state(self):
        server = create_server(0, api_key="k" * 32)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=5)
            conn.request("GET", "/v1/metrics", headers={"Host": "attacker.example",
                                                        "Authorization": "Bearer " + "k" * 32})
            response = conn.getresponse()
            self.assertEqual(response.status, 403)
            self.assertEqual(json.loads(response.read()), {"error": "forbidden_host"})
        finally:
            server.shutdown()
            server.server_close()

    def test_early_rejection_reads_the_body_first(self):
        """Regression (2026-11-07, Windows): replying 403/401/413 without reading the request
        body made Windows reset the connection, so clients saw WinError 10053 instead of the
        response. The server must drain the body before answering."""
        body = b'{"premises": ["p"], "conclusion": "p"}' + b" " * 20000
        head = (b"POST /v1/verify-logic HTTP/1.1\r\nHost: attacker.example\r\nContent-Type: application/json\r\n"
                b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n")
        with socket.create_connection(("127.0.0.1", self.port), timeout=5) as sock:
            sock.sendall(head + body[:1000])
            sock.settimeout(0.5)
            with self.assertRaises(socket.timeout):            # still reading the body, not replying yet
                sock.recv(1)
            sock.settimeout(5)
            sock.sendall(body[1000:])
            reply = b""
            while b"\r\n\r\n" not in reply:
                reply += sock.recv(4096)
        self.assertTrue(reply.startswith(b"HTTP/1.1 403"), reply[:40])

    def test_lying_content_length_does_not_hang_the_server(self):
        from cpu_platform import server as srv
        original = srv.DRAIN_TIMEOUT_S
        srv.DRAIN_TIMEOUT_S = 0.5
        self.addCleanup(setattr, srv, "DRAIN_TIMEOUT_S", original)
        head = (b"POST /v1/verify-logic HTTP/1.1\r\nHost: attacker.example\r\nContent-Type: application/json\r\n"
                b"Content-Length: 100000\r\n\r\n{}")
        start = time.monotonic()
        with socket.create_connection(("127.0.0.1", self.port), timeout=5) as sock:
            sock.sendall(head)                                  # promises 100 kB, sends 2 bytes
            reply = sock.recv(4096)
        self.assertTrue(reply.startswith(b"HTTP/1.1 403"), reply[:40])
        self.assertLess(time.monotonic() - start, 4)
        self.assertEqual(self.request("GET", "/healthz")[0], 200)   # server still serving

    def test_unsupported_methods_get_json_405(self):
        for method in ("PUT", "PATCH", "OPTIONS"):
            with self.subTest(method=method):
                self.assertEqual(self.request(method, "/v1/ask"), (405, {"error": "method_not_allowed"}))
        self.assertEqual(self.request("PUT", "/v1/ask", headers={"Host": "attacker.example"})[0], 403)


class ExtraAllowedHosts(ServerCase):
    server_kwargs = {"allowed_hosts": ("noema.internal",)}

    def test_operator_named_host_accepted(self):
        status, _ = self.request("POST", "/v1/verify-logic", LOGIC,
                                 {"Host": "noema.internal", "Origin": "https://NOEMA.internal"})
        self.assertEqual(status, 200)
        self.assertEqual(self.request("GET", "/healthz", headers={"Host": "other.internal"})[0], 403)

    def test_invalid_allowed_hosts_refused(self):
        for bad in ("", "evil.com/path", "*", "a b"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                create_server(0, allowed_hosts=(bad,))


class CatchAllErrors(ServerCase):
    server_kwargs = {"max_inflight": 1}

    def test_internal_error_returns_json_500_and_releases_capacity(self):
        secret_question = "SECRET-QUESTION-TEXT 2+2"
        original = self.server.engine.reason
        self.server.engine.reason = mock.Mock(side_effect=RuntimeError("internal bug"))
        with self.assertLogs("noema.server", level="ERROR") as logs:
            status, body = self.request("POST", "/v1/reason",
                                        json.dumps({"question": secret_question}))
        self.assertEqual(status, 500)
        self.assertEqual(body["error"], "internal_error")
        self.assertFalse(body["retryable"])
        self.assertRegex(body["request_id"], r"^[0-9a-f]{16}$")
        logged = "\n".join(logs.output)
        self.assertIn("RuntimeError: internal bug", logged)
        self.assertIn(body["request_id"], logged)
        self.assertNotIn("SECRET-QUESTION-TEXT", logged)
        # max_inflight=1: if the slot leaked, this would be 429.
        self.server.engine.reason = original
        self.assertEqual(self.request("POST", "/v1/verify-logic", LOGIC)[0], 200)
        # The 200 is written before the handler's finally block decrements the counter.
        for _ in range(100):
            if self.server.metrics.snapshot()["inflight"] == 0:
                break
            time.sleep(0.01)
        self.assertEqual(self.server.metrics.snapshot()["inflight"], 0)

    def test_internal_error_on_get_route(self):
        self.server.metrics.snapshot = mock.Mock(side_effect=KeyError("boom"))
        with self.assertLogs("noema.server", level="ERROR"):
            status, body = self.request("GET", "/v1/metrics")
        self.assertEqual((status, body["error"]), (500, "internal_error"))

    def test_error_after_response_started_is_logged_not_resent(self):
        # metrics.end runs in a finally block after the 200 has been written. A failure
        # there must be logged without appending a second response to the stream.
        self.server.metrics.end = mock.Mock(side_effect=RuntimeError("metrics bug"))
        with self.assertLogs("noema.server", level="ERROR") as logs:
            reply = self.raw(b"POST /v1/verify-logic HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                             b"Content-Type: application/json\r\nContent-Length: "
                             + str(len(LOGIC)).encode() + b"\r\n\r\n" + LOGIC.encode())
        self.assertTrue(reply.startswith(b"HTTP/1.1 200"), reply[:40])
        self.assertEqual(reply.count(b"HTTP/1.1 "), 1)
        self.assertIn("RuntimeError: metrics bug", "\n".join(logs.output))


class ModelServerApiKey(unittest.TestCase):
    KEY = "test-llama-key-0123456789"

    def setUp(self):
        self.fake = FakeLlama("Qwen3-0.6B-fake", accuracy=1.0, api_key=self.KEY,
                              model_path=None)

    def tearDown(self):
        self.fake.close()

    def test_client_without_key_is_refused(self):
        with mock.patch.dict(os.environ, {LLAMA_API_KEY_ENV: ""}):
            # M1.4: a 4xx is classified as a non-retryable rejected request (was a raw HTTPError).
            with self.assertRaises(ModelRejectedRequest) as caught:
                LocalLlama(self.fake.url)
        self.assertEqual(caught.exception.status, 401)
        self.assertFalse(caught.exception.retryable)
        self.assertIsInstance(caught.exception.__cause__, HTTPError)

    def test_client_with_key_connects_and_chats(self):
        with mock.patch.dict(os.environ, {LLAMA_API_KEY_ENV: self.KEY}):
            client = LocalLlama(self.fake.url)
            completion = client.chat("Answer.", "What is 2+3?", max_tokens=32)
        self.assertEqual(client.model, "Qwen3-0.6B-fake")
        self.assertIsInstance(completion.content, str)
        self.assertEqual(len(self.fake.requests), 1)

    def test_lazy_model_ready_and_props_send_key(self):
        with mock.patch.dict(os.environ, {LLAMA_API_KEY_ENV: self.KEY}):
            self.assertTrue(LazyLocalModel(self.fake.url).ready())
        with mock.patch.dict(os.environ, {LLAMA_API_KEY_ENV: "wrong"}):
            self.assertFalse(LazyLocalModel(self.fake.url).ready())
        self.fake.model_path = "/models/fake.gguf"
        with mock.patch.dict(os.environ, {LLAMA_API_KEY_ENV: self.KEY}):
            self.assertEqual(server_props(self.fake.url)["model_path"], "/models/fake.gguf")
        with mock.patch.dict(os.environ, {LLAMA_API_KEY_ENV: ""}):
            self.assertIsNone(server_props(self.fake.url))

    def test_launcher_reports_enforced(self):
        port = int(self.fake.url.rsplit(":", 1)[1])
        self.assertEqual(launch.api_key_status(port, True), "enforced")
        self.assertEqual(launch.api_key_status(port, False), "not configured")


class LauncherKeyStatus(unittest.TestCase):
    def _server(self, code):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length") or 0))   # see fake_llama.do_POST
                self.send_response(code)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")

        httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        return httpd.server_port

    def test_server_ignoring_key_is_flagged(self):
        # A build that ignores LLAMA_API_KEY answers the unauthenticated probe with 200 or 400.
        self.assertEqual(launch.api_key_status(self._server(400), True), "NOT ENFORCED")
        self.assertEqual(launch.api_key_status(self._server(200), True), "NOT ENFORCED")

    def test_unreachable_server_is_not_checked(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        self.assertEqual(launch.api_key_status(port, True, timeout=1), "NOT CHECKED")

    def test_key_never_on_command_line_or_in_manifest(self):
        secret = "do-not-leak-this-key-42"
        out = io.StringIO()
        with mock.patch.dict(os.environ, {LLAMA_API_KEY_ENV: secret}), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            code = launch.main(["--model", "qwen3-0.6b", "--dry-run"])
        self.assertEqual(code, 0)
        text = out.getvalue()
        self.assertNotIn(secret, text)
        self.assertNotIn("--api-key", text)
        self.assertIn('"api_key": "configured"', text)


if __name__ == "__main__":
    unittest.main()
