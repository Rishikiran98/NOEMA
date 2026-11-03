"""M1.3 contract tests: EngineRequest / EngineResult, the generated JSON Schema, the v1
adapters, and that the existing /v1 endpoints are unchanged.

The optional ``jsonschema`` package (not a NOEMA dependency) cross-checks that the
generated schema and the standard-library decoder agree; those checks skip without it.
"""
import copy
import http.client
import json
import threading
import unittest
from pathlib import Path

from cpu_first.core import Completion
from cpu_platform import __version__, contract as c
from cpu_platform.general import GeneralReasoner
from cpu_platform.server import create_server

try:
    import jsonschema
except ImportError:  # optional cross-check only
    jsonschema = None

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "docs" / "engine-contract-v1.schema.json"


def good_result() -> dict:
    """A result exercising every section, valid under the contract."""
    return {
        "contract_version": "1.0", "request_id": "req-1", "task": "repo_fix", "status": "ok",
        "engine_version": "0.10.0", "answer": "Fix the off-by-one in range().",
        "checked_claims": [{"claim": "tests/test_x.py passes after the patch", "result": "verified",
                            "checker": "unittest", "scope": "the named test file passed in the sandbox copy",
                            "evidence_ids": ["run-1"], "notes": []}],
        "unverified_claims": [{"claim": "The bug is caused by an off-by-one.",
                               "reason": "model interpretation; not checked", "evidence_ids": ["span-1"]}],
        "evidence": [{"id": "span-1", "kind": "file_span", "source": "src/x.py", "start_line": 10,
                      "end_line": 14, "sha256": "a" * 64, "excerpt": "for i in range(n - 1):",
                      "retrieved_by": "bm25"}],
        "executions": [{"id": "run-1", "command": ["python", "-m", "unittest", "tests.test_x"],
                        "backend": "dev", "security_boundary": False, "network": False, "exit_code": 0,
                        "timed_out": False, "duration_s": 1.5, "stdout_tail": "OK", "stderr_tail": "",
                        "output_truncated": False}],
        "cost": {"model_calls": 3, "prompt_tokens": 900, "completion_tokens": 300,
                 "model_wall_seconds": 40.0, "total_wall_seconds": 45.5, "cpu_seconds": 210.0,
                 "peak_rss_bytes": None},
        "not_run": [{"item": "cost.peak_rss_bytes", "reason": "psutil not installed"},
                    {"item": "full test suite", "reason": "only the targeted test ran"}],
        "model": {"id": "qwen3-coder-30b-a3b", "sha256": None},
        "error": None,
    }


class Stub:
    model = "stub-model"

    def __init__(self, text):
        self.text = text

    def chat(self, *args, **kwargs):
        return Completion(self.text, 0.25, 10, 20, None)


class GeneratedSchema(unittest.TestCase):
    def test_committed_schema_matches_dataclasses(self):
        self.assertEqual(SCHEMA_PATH.read_text(encoding="utf-8"), c.schema_text(),
                         "regenerate: python -m cpu_platform.contract --write docs/engine-contract-v1.schema.json")

    def test_check_command_detects_stale_schema(self):
        import contextlib, io, tempfile
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            stale = Path(tmp) / "s.json"
            stale.write_text("{}\n", encoding="utf-8")
            self.assertEqual(c.main(["--check", str(stale)]), 1)
            self.assertEqual(c.main(["--check", str(SCHEMA_PATH)]), 0)

    def test_every_object_is_closed_and_has_required_list(self):
        for name, definition in c.json_schema()["$defs"].items():
            self.assertFalse(definition["additionalProperties"], name)
            self.assertTrue(set(definition["required"]) <= set(definition["properties"]), name)
        result = c.json_schema()["$defs"]["EngineResult"]
        self.assertNotIn("verified", result["properties"])  # no top-level verified flag, by design

    @unittest.skipIf(jsonschema is None, "jsonschema not installed")
    def test_schema_is_valid_and_agrees_with_decoder(self):
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator.check_schema(schema)
        validator = jsonschema.Draft202012Validator(schema)
        ok_request = {"question": "What does parse() return?", "task": "repo_ask", "repo": {"path": "/r"}}
        self.assertEqual(list(validator.iter_errors(ok_request)), [])
        self.assertEqual(list(validator.iter_errors(good_result())), [])
        c.request_from_dict(ok_request)
        c.result_from_dict(good_result())
        structural_violations = [
            {**ok_request, "extra": 1},
            {**ok_request, "question": ""},
            {**ok_request, "task": "shell"},
            {**ok_request, "budget": {"max_model_calls": 0}},
            {**ok_request, "budget": {"max_model_calls": 1.5}},
            {**ok_request, "budget": {"max_wall_seconds": 0}},
            {**ok_request, "client_request_id": "has spaces"},
            {**ok_request, "repo": {"path": "/r", "branch": "main"}},
        ]
        for bad in structural_violations:
            with self.subTest(bad=bad):
                self.assertTrue(list(validator.iter_errors(bad)))
                with self.assertRaises(c.ContractError):
                    c.request_from_dict(bad)
        for mutate in (lambda r: r["checked_claims"][0].update(checker=""),
                       lambda r: r["checked_claims"][0].update(evidence_ids=[]),
                       lambda r: r["checked_claims"][0].update(result="probably"),
                       lambda r: r["evidence"][0].update(sha256="xyz"),
                       lambda r: r["executions"][0].update(exit_code="0"),
                       lambda r: r.update(status="done"),
                       lambda r: r.update(verified=True)):
            bad = good_result()
            mutate(bad)
            with self.subTest(bad=bad):
                self.assertTrue(list(validator.iter_errors(bad)))
                with self.assertRaises(c.ContractError):
                    c.result_from_dict(bad)


class Requests(unittest.TestCase):
    def test_defaults(self):
        request = c.request_from_dict({"question": "Why?"})
        self.assertEqual((request.task, request.contract_version, request.allow_execution),
                         ("ask", "1.0", False))
        self.assertEqual(request.budget, c.Budget(2, 1024, 600.0))
        self.assertIsNone(request.thinking)

    def test_cross_field_rules(self):
        cases = {
            "repo: required for task 'repo_ask'": {"question": "q", "task": "repo_ask"},
            "repo: not allowed for task 'ask'": {"question": "q", "repo": {"path": "."}},
            "allow_execution: only task 'repo_fix' may execute code":
                {"question": "q", "task": "repo_ask", "repo": {"path": "."}, "allow_execution": True},
            "question: must not be blank": {"question": "   \n"},
        }
        for message, data in cases.items():
            with self.subTest(message=message), self.assertRaises(c.ContractError) as caught:
                c.request_from_dict(data)
            self.assertIn(message, caught.exception.problems)
        ok = c.request_from_dict({"question": "q", "task": "repo_fix", "repo": {"path": "."},
                                  "allow_execution": True})
        self.assertTrue(ok.allow_execution)

    def test_type_confusion_rejected(self):
        for data in ({"question": "q", "allow_execution": 1},       # int is not bool
                     {"question": "q", "budget": {"max_output_tokens": True}},  # bool is not int
                     {"question": "q", "budget": {"max_wall_seconds": float("nan")}},
                     {"question": "q", "budget": {"max_wall_seconds": float("inf")}},
                     {"question": 42},
                     {"question": "q", "contract_version": "2.0"},
                     {"question": "q", "contract_version": 1.0},
                     ["question", "q"], None, "q"):
            with self.subTest(data=data), self.assertRaises(c.ContractError):
                c.request_from_dict(data)

    def test_size_limits(self):
        c.request_from_dict({"question": "x" * c.MAX_QUESTION_CHARS})
        with self.assertRaises(c.ContractError):
            c.request_from_dict({"question": "x" * (c.MAX_QUESTION_CHARS + 1)})

    def test_every_problem_reported_at_once(self):
        with self.assertRaises(c.ContractError) as caught:
            c.request_from_dict({"question": "", "task": "x", "a": 1, "b": 2})
        self.assertEqual(len(caught.exception.problems), 4)

    def test_requests_are_always_strict(self):
        with self.assertRaises(c.ContractError):
            c.request_from_dict({"question": "q", "future_field": True})


class Results(unittest.TestCase):
    def test_round_trip(self):
        result = c.result_from_dict(good_result())
        self.assertEqual(json.loads(result.to_json()), good_result())
        self.assertEqual(c.result_from_dict(json.loads(result.to_json())), result)

    def invalid(self, mutate, expected):
        data = good_result()
        mutate(data)
        with self.assertRaises(c.ContractError) as caught:
            c.result_from_dict(data)
        self.assertTrue(any(expected in p for p in caught.exception.problems), caught.exception.problems)

    def test_dev_backend_can_never_claim_isolation(self):
        self.invalid(lambda r: r["executions"][0].update(security_boundary=True),
                     "backend 'dev' is not a security boundary")
        data = good_result()
        data["executions"][0].update(backend="container", security_boundary=True)
        c.result_from_dict(data)

    def test_claims_must_cite_known_evidence(self):
        self.invalid(lambda r: r["checked_claims"][0].update(evidence_ids=["invented"]), "unknown evidence ids")
        self.invalid(lambda r: r["unverified_claims"][0].update(evidence_ids=["nope"]), "unknown evidence ids")

    def test_ids_unique_across_evidence_and_executions(self):
        self.invalid(lambda r: r["executions"][0].update(id="span-1"), "duplicated")

    def test_status_error_consistency(self):
        self.invalid(lambda r: r.update(status="error"), "error: required exactly")
        self.invalid(lambda r: r.update(error={"code": "x", "message": "m", "retryable": False}),
                     "error: required exactly")
        self.invalid(lambda r: r.update(answer=None), "answer: required when status is 'ok'")
        data = good_result()
        data.update(status="error", answer=None,
                    error={"code": "model_unavailable", "message": "timeout", "retryable": True})
        c.result_from_dict(data)

    def test_unmeasured_cost_must_be_declared_not_run(self):
        self.invalid(lambda r: r["not_run"].pop(0), "unmeasured cost.peak_rss_bytes")
        self.invalid(lambda r: r["cost"].update(cpu_seconds=None), "unmeasured cost.cpu_seconds")
        self.assertEqual([n.item for n in c.not_measured(c.Cost())], ["cost.cpu_seconds", "cost.peak_rss_bytes"])

    def test_file_spans_need_valid_lines(self):
        self.invalid(lambda r: r["evidence"][0].update(start_line=None), "file_span needs")
        self.invalid(lambda r: r["evidence"][0].update(start_line=20), "end_line before start_line")
        self.invalid(lambda r: r["evidence"][0].update(start_line=0), "below 1")

    def test_refuted_claims_are_first_class(self):
        data = good_result()
        data["checked_claims"][0]["result"] = "refuted"
        self.assertEqual(c.result_from_dict(data).checked_claims[0].result, "refuted")

    def test_forward_compatible_reading(self):
        newer = {**good_result(), "contract_version": "1.3", "patch": {"diff": "..."}}
        with self.assertRaises(c.ContractError):
            c.result_from_dict(newer)
        self.assertEqual(c.result_from_dict(newer, strict=False).contract_version, "1.0")
        with self.assertRaises(c.ContractError):
            c.result_from_dict({**newer, "contract_version": "2.0"}, strict=False)

    def test_producers_cannot_emit_invalid_results(self):
        result = c.result_from_dict(good_result())
        result.executions[0].security_boundary = True
        with self.assertRaises(c.ContractError):
            c.validate_result(result)


class V1Adapters(unittest.TestCase):
    QUESTION = "A box holds 6 rows of 12 eggs. 50 are sold. How many remain?"

    def test_model_answer_is_unverified_and_each_equation_checked_separately(self):
        reasoner = GeneralReasoner(Stub("First 6 * 12 = 72, then 72 - 50 = 23. So the answer is 23."))
        legacy = reasoner.ask(self.QUESTION, "math", "direct")
        result = c.from_v1_ask(self.QUESTION, legacy, request_id="r1", engine_version=__version__,
                               model_id="stub-model")
        self.assertEqual(result.status, "ok")
        self.assertEqual([u.claim for u in result.unverified_claims], [legacy["answer"]])
        self.assertEqual([(k.claim, k.result) for k in result.checked_claims],
                         [("6 * 12 = 72", "verified"), ("72 - 50 = 23", "refuted")])
        self.assertIn("exact value is 22", result.checked_claims[1].notes)
        for claim in result.checked_claims:
            self.assertIn("does not show", claim.scope)
            self.assertNotIn(legacy["answer"], claim.claim)  # never a claim about the answer
        self.assertEqual(result.cost.model_calls, 1)
        self.assertIn("cost.cpu_seconds", [n.item for n in result.not_run])

    def test_correct_arithmetic_still_leaves_answer_unverified(self):
        reasoner = GeneralReasoner(Stub("6 * 12 = 72 and 72 - 50 = 22, so 22 remain."))
        legacy = reasoner.ask(self.QUESTION, "math", "direct")
        result = c.from_v1_ask(self.QUESTION, legacy, request_id="r2", engine_version=__version__)
        self.assertTrue(all(k.result == "verified" for k in result.checked_claims))
        self.assertEqual(len(result.unverified_claims), 1)

    def test_exact_arithmetic_is_one_verified_claim(self):
        legacy = GeneralReasoner(Stub("unused")).ask("12 * 7", "auto", "direct")
        result = c.from_v1_ask("12 * 7", legacy, request_id="r3", engine_version=__version__)
        self.assertEqual((result.answer, len(result.checked_claims), result.unverified_claims), ("84", 1, []))
        self.assertEqual(result.checked_claims[0].checker, "exact_arithmetic")
        self.assertEqual(result.cost.model_calls, 0)

    def test_code_and_factual_answers_list_what_did_not_run(self):
        for domain, item in (("code", "code execution"), ("factual", "source retrieval")):
            legacy = GeneralReasoner(Stub("def f(): return 1")).ask("Write f", domain, "direct")
            result = c.from_v1_ask("Write f", legacy, request_id="r4", engine_version=__version__)
            with self.subTest(domain=domain):
                self.assertIn(item, [n.item for n in result.not_run])
                self.assertEqual(result.checked_claims, [])

    def test_empty_model_answer_is_incomplete(self):
        legacy = GeneralReasoner(Stub("")).ask("Explain recursion", "general", "direct")
        result = c.from_v1_ask("Explain recursion", legacy, request_id="r5", engine_version=__version__)
        self.assertEqual((result.status, result.answer, result.unverified_claims), ("incomplete", None, []))

    def test_legacy_dict_is_not_modified(self):
        legacy = GeneralReasoner(Stub("2 + 2 = 4")).ask("What is two plus two, explained?", "math", "direct")
        before = copy.deepcopy(legacy)
        c.from_v1_ask("q", legacy, request_id="r6", engine_version=__version__)
        self.assertEqual(legacy, before)


class V1EndpointsUnchanged(unittest.TestCase):
    """The contract is additive: /v1 responses keep their exact shape."""

    def setUp(self):
        self.server = create_server(0)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def post(self, path, body):
        conn = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        conn.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
        response = conn.getresponse()
        return response.status, json.loads(response.read())

    def test_ask_and_logic_shapes(self):
        status, ask = self.post("/v1/ask", {"question": "12 * 7"})
        self.assertEqual(status, 200)
        self.assertEqual(set(ask), {"status", "answer", "domain", "route", "verified", "verification_scope",
                                    "checks", "usage", "request_id", "version"})
        status, logic = self.post("/v1/verify-logic", {"premises": ["p -> q", "p"], "conclusion": "q"})
        self.assertEqual(status, 200)
        self.assertEqual(set(logic), {"status", "answer", "route", "verified", "verification_scope",
                                      "checks", "usage", "request_id", "version"})
        mapped = c.from_v1_logic(["p -> q", "p"], "q", logic, request_id=logic["request_id"],
                                 engine_version=logic["version"])
        self.assertEqual(mapped.checked_claims[0].checker, "propositional_logic")
        self.assertEqual(mapped.answer, "entailed")

    def test_no_v2_route_yet(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        conn.request("POST", "/v2/engine", "{}", {"Content-Type": "application/json"})
        self.assertEqual(conn.getresponse().status, 404)


if __name__ == "__main__":
    unittest.main()
