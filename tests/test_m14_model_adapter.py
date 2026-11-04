"""M1.4 tests: model-server error classification (audit G3) and model identity, context
and hash checks before serving (audit G5). Offline only, against fake llama.cpp servers."""
import hashlib
import http.client
import json
import socket
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from cpu_first.core import (Completion, LocalLlama, ModelError, ModelMismatch, ModelProtocolError,
                            ModelRejectedRequest, ModelTransportError, thinking_support)
from cpu_platform import model_adapter as ma
from cpu_platform.engine import LazyLocalModel, ReasoningEngine, UpstreamFailed, UpstreamUnavailable
from cpu_platform.server import check_model_at_startup, create_server
from tests.fake_llama import FakeLlama

CODER_FILE = "Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf"


def closed_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class ScriptedServer:
    """A model server that lists one model and answers chat with a scripted response."""

    def __init__(self, *, chat_status=200, chat_body=b'{"choices":[{"message":{"content":"ok"}}]}',
                 models_body=None, delay_s=0.0):
        script = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _raw(self, code, body):
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass  # the client timed out first (the timeout test)

            def do_GET(self):
                if self.path == "/v1/models":
                    return self._raw(200, script.models_body)
                self._raw(404, b"{}")

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                time.sleep(script.delay_s)
                self._raw(script.chat_status, script.chat_body)

        self.chat_status, self.chat_body, self.delay_s = chat_status, chat_body, delay_s
        self.models_body = models_body or b'{"data":[{"id":"Qwen3-0.6B-Q8_0.gguf"}]}'
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}"
        threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


class ErrorClassification(unittest.TestCase):
    def scripted(self, **kw):
        server = ScriptedServer(**kw)
        self.addCleanup(server.close)
        return server

    def test_connection_refused_is_retryable_transport(self):
        with self.assertRaises(ModelTransportError) as caught:
            LocalLlama(f"http://127.0.0.1:{closed_port()}")
        self.assertTrue(caught.exception.retryable)

    def test_http_5xx_is_retryable_transport(self):
        server = self.scripted(chat_status=503, chat_body=b'{"error":"loading model"}')
        with self.assertRaises(ModelTransportError):
            LocalLlama(server.url).chat("s", "q")

    def test_http_4xx_is_rejected_not_retryable(self):
        server = self.scripted(chat_status=400, chat_body=b'{"error":"context length exceeded"}')
        with self.assertRaises(ModelRejectedRequest) as caught:
            LocalLlama(server.url).chat("s", "q")
        self.assertEqual((caught.exception.status, caught.exception.retryable), (400, False))

    def test_timeout_is_retryable_transport(self):
        server = self.scripted(delay_s=2.0)
        client = LocalLlama(server.url, timeout=1)
        with self.assertRaises(ModelTransportError):
            client.chat("s", "q")

    def test_malformed_responses_are_protocol_errors(self):
        for body in (b"not json", b"[1,2]", b'{"choices":[]}', b'{"choices":[{"message":null}]}',
                     b'{"choices":[{"message":{"content":5}}]}', b'{"choices":[{"message":{"content":"x"}}],"usage":3}'):
            with self.subTest(body=body):
                server = self.scripted(chat_body=body)
                with self.assertRaises(ModelProtocolError) as caught:
                    LocalLlama(server.url).chat("s", "q")
                self.assertFalse(caught.exception.retryable)
        for models in (b'{"data":[]}', b'{"data":"x"}', b'{"data":[{"name":"m"}]}', b'{}'):
            with self.subTest(models=models):
                server = self.scripted(models_body=models)
                with self.assertRaises(ModelProtocolError):
                    LocalLlama(server.url)

    def test_all_model_errors_remain_runtime_errors(self):
        for cls in (ModelError, ModelTransportError, ModelRejectedRequest, ModelProtocolError, ModelMismatch):
            self.assertTrue(issubclass(cls, RuntimeError), cls)


class ThinkingRequests(unittest.TestCase):
    def test_support_by_family(self):
        self.assertIs(thinking_support("Qwen3-8B-Q4_K_M.gguf"), True)
        self.assertIs(thinking_support(CODER_FILE), False)
        self.assertIs(thinking_support("qwen2.5-coder-7b-instruct-q4_k_m.gguf"), False)
        self.assertIsNone(thinking_support("Llama-3.1-8B-Instruct"))

    def test_thinking_refused_for_non_thinking_model(self):
        fake = FakeLlama(CODER_FILE, accuracy=1.0, think_capable=False)
        self.addCleanup(fake.close)
        client = LocalLlama(fake.url)
        with self.assertRaisesRegex(ValueError, "does not support per-request thinking"):
            client.chat("s", "q", thinking=True)
        self.assertEqual(fake.requests, [])  # refused before anything was sent
        client.chat("s", "q", thinking=False)

    def test_unknown_family_passes_thinking_through(self):
        fake = FakeLlama("Llama-3.1-8B-Instruct", accuracy=1.0)
        self.addCleanup(fake.close)
        LocalLlama(fake.url).chat("s", "q", thinking=True)
        self.assertEqual(fake.requests[-1]["chat_template_kwargs"], {"enable_thinking": True})


class Stub:
    def __init__(self, exc=None):
        self.exc = exc

    def chat(self, *args, **kwargs):
        if self.exc:
            raise self.exc
        return Completion("Paris is the capital of France.", 0.1, 5, 7, None)


class ApiMapping(unittest.TestCase):
    def serve(self, engine, **kw):
        server = create_server(0, engine=engine, **kw)
        threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return server

    def post(self, server, path, body):
        conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)
        conn.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
        response = conn.getresponse()
        return response.status, json.loads(response.read())

    def test_each_model_error_maps_to_its_status(self):
        cases = [
            (ModelTransportError("down"), 503, {"error": "local_model_unavailable", "retryable": True}),
            (ModelRejectedRequest("HTTP 400", 400), 502, {"error": "local_model_rejected_request", "retryable": False}),
            (ModelProtocolError("bad json"), 502, {"error": "local_model_bad_response", "retryable": False}),
            (ModelMismatch(["wrong model"]), 503, {"error": "local_model_mismatch", "retryable": False}),
        ]
        for exc, status, body in cases:
            server = self.serve(ReasoningEngine(Stub(exc)))
            for path, payload in (("/v1/ask", {"question": "Capital of France?"}),
                                  ("/v1/reason", {"question": "Tom has 3 apples and buys 4. How many?",
                                                  "mode": "baseline"})):
                with self.subTest(exc=type(exc).__name__, path=path):
                    self.assertEqual(self.post(server, path, payload), (status, body))

    def test_programming_error_is_500_not_retryable_outage(self):
        for exc in (KeyError("bug"), AttributeError("bug"), TypeError("bug")):
            server = self.serve(ReasoningEngine(Stub(exc)))
            with self.subTest(exc=type(exc).__name__), self.assertLogs("noema.server", "ERROR"):
                status, body = self.post(server, "/v1/ask", {"question": "Capital of France?"})
            self.assertEqual((status, body["error"], body["retryable"]), (500, "internal_error", False))

    def test_thinking_request_to_coder_model_is_400(self):
        fake = FakeLlama(CODER_FILE, accuracy=1.0, think_capable=False)
        self.addCleanup(fake.close)
        server = self.serve(ReasoningEngine(LazyLocalModel(fake.url)))
        status, body = self.post(server, "/v1/ask", {"question": "Explain recursion", "thinking": True})
        self.assertEqual(status, 400)
        self.assertIn("does not support per-request thinking", body["detail"])

    def test_real_unreachable_server_is_503_retryable(self):
        server = self.serve(ReasoningEngine(LazyLocalModel(f"http://127.0.0.1:{closed_port()}")))
        self.assertEqual(self.post(server, "/v1/ask", {"question": "Capital of France?"}),
                         (503, {"error": "local_model_unavailable", "retryable": True}))


class Expectations(unittest.TestCase):
    def test_preset_expectations(self):
        coder = ma.expectation_for_preset("qwen3-coder-30b-a3b")
        self.assertEqual(coder.name_fragments, ("Qwen3-Coder-30B-A3B-Instruct", "Q4_K_M"))
        self.assertEqual((coder.min_ctx, coder.thinking), (16384, False))
        small = ma.expectation_for_preset("qwen3-0.6b")
        self.assertEqual((small.name_fragments, small.min_ctx, small.thinking), (("Qwen3-0.6B", "Q8_0"), None, True))
        with self.assertRaises(ValueError):
            ma.expectation_for_preset("gpt-5")

    def test_parse_expectation(self):
        self.assertEqual(ma.parse_expectation("qwen3-coder-30b-a3b", min_ctx=8192).min_ctx, 8192)
        custom = ma.parse_expectation("mistral, Q5_K_M", min_ctx=4096)
        self.assertEqual((custom.name_fragments, custom.min_ctx, custom.thinking), (("mistral", "Q5_K_M"), 4096, None))
        for bad in ("", " , "):
            with self.assertRaises(ValueError):
                ma.parse_expectation(bad)
        for kwargs in ({"min_ctx": 0}, {"min_ctx": True}, {"sha256": "ABC"}, {"name_fragments": ("",)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ma.ModelExpectation(**kwargs)


class Verify(unittest.TestCase):
    def fake(self, model_id=CODER_FILE, **kw):
        kw.setdefault("serve_props", True)
        fake = FakeLlama(model_id, accuracy=1.0, think_capable=False, **kw)
        self.addCleanup(fake.close)
        return fake

    def problems(self, fake, expect):
        return ma.problems(ma.probe(fake.url, hash_file=expect.sha256 is not None), expect)

    def test_matching_coder_passes(self):
        fake = self.fake(n_ctx=16384)
        identity = ma.verify(fake.url, ma.expectation_for_preset("qwen3-coder-30b-a3b"))
        self.assertEqual((identity.id, identity.n_ctx, identity.thinking_supported), (CODER_FILE, 16384, False))

    def test_context_too_small(self):
        found = self.problems(self.fake(n_ctx=4096), ma.expectation_for_preset("qwen3-coder-30b-a3b"))
        self.assertEqual(len(found), 1)
        self.assertIn("relaunch with --ctx-size 16384", found[0])

    def test_wrong_model(self):
        found = self.problems(self.fake("Qwen3-8B-Q4_K_M.gguf", n_ctx=16384),
                              ma.expectation_for_preset("qwen3-coder-30b-a3b"))
        self.assertTrue(any("missing 'Qwen3-Coder-30B-A3B-Instruct'" in p for p in found), found)
        self.assertTrue(any("expected thinking support False" in p for p in found), found)

    def test_wrong_quantization(self):
        found = self.problems(self.fake("Qwen3-Coder-30B-A3B-Instruct-Q8_0.gguf", n_ctx=16384),
                              ma.expectation_for_preset("qwen3-coder-30b-a3b"))
        self.assertEqual(found, ["model 'Qwen3-Coder-30B-A3B-Instruct-Q8_0.gguf' does not match expected "
                                 "'qwen3-coder-30b-a3b' (missing 'Q4_K_M')"])

    def test_alias_id_matched_through_model_path(self):
        fake = self.fake("default-alias", n_ctx=16384,
                         model_path="/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf")
        self.assertEqual(self.problems(fake, ma.expectation_for_preset("qwen2.5-coder-7b")), [])

    def test_several_models_listed(self):
        found = self.problems(self.fake(n_ctx=16384, extra_models=("other.gguf",)),
                              ma.expectation_for_preset("qwen3-coder-30b-a3b"))
        self.assertTrue(any("lists 2 models" in p for p in found), found)

    def test_unknown_context_fails_closed(self):
        found = self.problems(self.fake(serve_props=False), ma.ModelExpectation(min_ctx=2048))
        self.assertEqual(found, ["context size unknown (server /props did not report n_ctx); expected at least 2048"])
        self.assertEqual(self.problems(self.fake(serve_props=False), ma.ModelExpectation()), [])

    def test_sha256_checked_against_loaded_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / CODER_FILE
            path.write_bytes(b"fake gguf weights")
            digest = hashlib.sha256(b"fake gguf weights").hexdigest()
            fake = self.fake(n_ctx=16384, model_path=str(path))
            self.assertEqual(self.problems(fake, ma.ModelExpectation(sha256=digest)), [])
            found = self.problems(fake, ma.ModelExpectation(sha256="0" * 64))
            self.assertTrue(found and "differs from the expected" in found[0], found)
        missing = self.fake(n_ctx=16384, model_path="/nonexistent/model.gguf")
        found = self.problems(missing, ma.ModelExpectation(sha256=digest))
        self.assertTrue(found and "could not be computed" in found[0], found)

    def test_props_auth_error_is_not_swallowed(self):
        fake = FakeLlama(CODER_FILE, accuracy=1.0, api_key="k", serve_props=True)
        self.addCleanup(fake.close)
        import os
        from unittest import mock
        with mock.patch.dict(os.environ, {"LLAMA_API_KEY": ""}):
            with self.assertRaises(ModelRejectedRequest):
                ma.probe(fake.url)


class LazyModelWithExpectation(unittest.TestCase):
    def setUp(self):
        self.good = FakeLlama(CODER_FILE, accuracy=1.0, think_capable=False, n_ctx=16384, serve_props=True)
        self.small = FakeLlama(CODER_FILE, accuracy=1.0, think_capable=False, n_ctx=4096, serve_props=True)
        self.addCleanup(self.good.close)
        self.addCleanup(self.small.close)
        self.expect = ma.expectation_for_preset("qwen3-coder-30b-a3b")

    def get(self, server, path):
        conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)
        conn.request("GET", path)
        response = conn.getresponse()
        return response.status, json.loads(response.read())

    def test_mismatch_blocks_every_request_and_is_reported(self):
        model = LazyLocalModel(self.small.url, expectation=self.expect)
        with self.assertRaises(UpstreamFailed) as caught:
            model.chat("s", "q")
        self.assertEqual((caught.exception.code, caught.exception.http_status), ("local_model_mismatch", 503))
        self.assertEqual(self.small.requests, [])  # nothing reached the wrong model
        self.assertIn("relaunch with --ctx-size 16384", model.last_problems[0])
        server = create_server(0, engine=ReasoningEngine(model))
        threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
        try:
            status, body = self.get(server, "/readyz")
            self.assertEqual((status, body["model_ready"], body["model_expected"]), (503, False, "qwen3-coder-30b-a3b"))
            self.assertTrue(body["model_problems"])
        finally:
            server.shutdown()
            server.server_close()

    def test_match_serves_and_reports_identity(self):
        model = LazyLocalModel(self.good.url, expectation=self.expect)
        model.chat("s", "q")
        self.assertEqual(len(self.good.requests), 1)
        server = create_server(0, engine=ReasoningEngine(model))
        threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
        try:
            status, body = self.get(server, "/readyz")
            self.assertEqual(status, 200)
            self.assertEqual(body["model"], {"id": CODER_FILE, "n_ctx": 16384})
            self.assertEqual(body["model_problems"], [])
        finally:
            server.shutdown()
            server.server_close()

    def test_no_expectation_keeps_v010_behaviour(self):
        model = LazyLocalModel(self.small.url)
        model.chat("s", "q")
        self.assertIsNone(model.identity)

    def test_startup_check_messages(self):
        self.assertTrue(check_model_at_startup(LazyLocalModel(self.good.url, expectation=self.expect))
                        .startswith("Model check passed: " + CODER_FILE))
        message = check_model_at_startup(LazyLocalModel(self.small.url, expectation=self.expect))
        self.assertTrue(message.startswith("ERROR: refusing to serve"), message)
        self.assertIn("--ctx-size 16384", message)
        message = check_model_at_startup(LazyLocalModel(f"http://127.0.0.1:{closed_port()}", expectation=self.expect))
        self.assertTrue(message.startswith("WARNING: model server not reachable yet"), message)


class JobsMapping(unittest.TestCase):
    def run_job(self, exc):
        with tempfile.TemporaryDirectory() as tmp:
            server = create_server(0, engine=ReasoningEngine(Stub(exc)), jobs_db=str(Path(tmp) / "jobs.db"))
            threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
            try:
                conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)
                conn.request("POST", "/v1/jobs", json.dumps({"question": "Secret question text?", "mode": "baseline"}),
                             {"Content-Type": "application/json"})
                job_id = json.loads(conn.getresponse().read())["id"]
                for _ in range(200):
                    conn = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)
                    conn.request("GET", f"/v1/jobs/{job_id}")
                    job = json.loads(conn.getresponse().read())
                    if job["status"] not in ("queued", "running"):
                        return job
                    time.sleep(0.02)
                self.fail("job did not finish")
            finally:
                server.shutdown()
                server.server_close()

    def test_job_errors_are_classified(self):
        cases = ((ModelTransportError("down"), "local_model_unavailable"),
                 (ModelMismatch(["wrong"]), "local_model_mismatch"),
                 (ModelRejectedRequest("HTTP 400", 400), "local_model_rejected_request"),
                 (KeyError("bug"), "internal_worker_error"))
        for exc, code in cases:
            with self.subTest(exc=type(exc).__name__):
                job = self.run_job(exc)
                self.assertEqual((job["status"], job["error"]), ("failed", code))
                self.assertNotIn("Secret question", json.dumps(job))


if __name__ == "__main__":
    unittest.main()
