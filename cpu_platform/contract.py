"""NOEMA Engine request/result contract, version 1.0 (roadmap M1.3).

One shape for every engine answer, so a caller (and an auditor) can always tell:

* what evidence was looked at (``evidence``),
* which specific claims a *named checker* verified or refuted (``checked_claims``),
* which claims nobody checked (``unverified_claims``),
* what was executed, by which backend, and whether that backend is a security
  boundary (``executions``),
* what it cost (``cost``), and
* what was NOT run or NOT measured (``not_run``).

There is deliberately no top-level ``verified`` flag. A model answer is unverified
unless a checker verified a specific claim, and even then only that claim, within the
stated scope (CLAUDE.md: "Correct arithmetic or passing tests never imply the
interpretation was correct").

The dataclasses are the single source of truth: ``from_dict`` validates against their
type hints, and ``json_schema()`` generates ``docs/engine-contract-v1.schema.json`` from
the same hints. Standard library only. Existing ``/v1`` endpoints are unchanged; the
``from_v1_*`` adapters show how their results map onto this contract.

Versioning: ``contract_version`` is ``"<major>.<minor>"``. A minor version only adds
optional fields; readers that must accept newer minors decode with ``strict=False``,
which ignores unknown fields. Requests are always decoded strictly.
"""
from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
import json
import math
import re
import types
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

CONTRACT_VERSION = "1.0"
SCHEMA_ID = "https://noema.local/schemas/engine-contract-v1.schema.json"

MAX_QUESTION_CHARS = 20_000
MAX_EXCERPT_CHARS = 4_000
MAX_OUTPUT_TAIL_CHARS = 16_000
ID_PATTERN = r"^[A-Za-z0-9_.:-]{1,64}$"

# Execution backends that are known NOT to isolate the host. The dev backend runs in a
# temporary copy of the repo with timeouts, which limits accidents but not attacks.
NON_BOUNDARY_BACKENDS = frozenset({"dev"})

Task = Literal["ask", "repo_ask", "repo_fix"]
REPO_TASKS = frozenset({"repo_ask", "repo_fix"})


class ContractError(ValueError):
    """The data does not satisfy the contract. ``problems`` lists every violation."""

    def __init__(self, problems: list[str]):
        self.problems = problems
        super().__init__("; ".join(problems))


def _f(default: Any = MISSING, *, factory: Any = MISSING, **constraints: Any):
    """A dataclass field whose JSON constraints (minLength, minimum, ...) live in metadata,
    so the decoder and the generated schema enforce the same rules."""
    kwargs: dict[str, Any] = {"metadata": constraints}
    if factory is not MISSING:
        kwargs["default_factory"] = factory
    elif default is not MISSING:
        kwargs["default"] = default
    return field(**kwargs)


# --------------------------------------------------------------------------- request

@dataclass
class RepoRef:
    path: str = _f(minLength=1, maxLength=4096, description="local repository root")
    revision: str | None = _f(None, maxLength=200, description="commit, branch or tag; null = working tree")


@dataclass
class Budget:
    max_model_calls: int = _f(2, minimum=1, maximum=64)
    max_output_tokens: int = _f(1024, minimum=1, maximum=32768)
    max_wall_seconds: float = _f(600.0, exclusiveMinimum=0, maximum=86400)


@dataclass
class EngineRequest:
    question: str = _f(minLength=1, maxLength=MAX_QUESTION_CHARS)
    task: Task = _f("ask", description="ask: no repository; repo_ask: grounded Q&A (M2.3); "
                                        "repo_fix: propose and test a patch (M3)")
    contract_version: Literal["1.0"] = "1.0"
    repo: RepoRef | None = None
    budget: Budget = _f(factory=Budget)
    allow_execution: bool = _f(False, description="permit sandboxed execution (repo_fix only)")
    thinking: bool | None = _f(None, description="null = the model adapter's default")
    client_request_id: str | None = _f(None, pattern=ID_PATTERN)

    def problems(self) -> list[str]:
        out = []
        if not self.question.strip():
            out.append("question: must not be blank")
        if self.task in REPO_TASKS and self.repo is None:
            out.append(f"repo: required for task {self.task!r}")
        if self.task == "ask" and self.repo is not None:
            out.append("repo: not allowed for task 'ask'")
        if self.allow_execution and self.task != "repo_fix":
            out.append("allow_execution: only task 'repo_fix' may execute code")
        return out


# --------------------------------------------------------------------------- result

@dataclass
class Evidence:
    id: str = _f(pattern=ID_PATTERN)
    kind: Literal["input", "model_output", "file_span", "command_output"] = _f()
    source: str = _f(minLength=1, maxLength=4096,
                     description="where it came from: 'question', a repo-relative path, an execution id")
    start_line: int | None = _f(None, minimum=1)
    end_line: int | None = _f(None, minimum=1)
    sha256: str | None = _f(None, pattern=r"^[0-9a-f]{64}$", description="SHA256 of the full span text")
    excerpt: str | None = _f(None, maxLength=MAX_EXCERPT_CHARS)
    retrieved_by: str | None = _f(None, maxLength=200, description="retriever or step that produced it")


@dataclass
class CheckedClaim:
    claim: str = _f(minLength=1, maxLength=4000)
    result: Literal["verified", "refuted"] = _f()
    checker: str = _f(minLength=1, maxLength=200, description="named checker, e.g. 'exact_arithmetic'")
    scope: str = _f(minLength=1, maxLength=2000, description="exactly what the check establishes")
    evidence_ids: list[str] = _f(minItems=1)
    notes: list[str] = _f(factory=list)


@dataclass
class UnverifiedClaim:
    claim: str = _f(minLength=1, maxLength=20000)
    reason: str = _f(minLength=1, maxLength=2000, description="why it was not checked")
    evidence_ids: list[str] = _f(factory=list)


@dataclass
class ExecutionRecord:
    id: str = _f(pattern=ID_PATTERN)
    command: list[str] = _f(minItems=1)
    backend: str = _f(minLength=1, maxLength=64, description="sandbox backend that ran it, e.g. 'dev', 'container'")
    security_boundary: bool = _f(description="true only for a backend that isolates the host; 'dev' never does")
    network: bool = _f()
    exit_code: int | None = _f(description="null when killed or not started")
    timed_out: bool = _f()
    duration_s: float = _f(minimum=0)
    stdout_tail: str = _f("", maxLength=MAX_OUTPUT_TAIL_CHARS)
    stderr_tail: str = _f("", maxLength=MAX_OUTPUT_TAIL_CHARS)
    output_truncated: bool = False


@dataclass
class Cost:
    model_calls: int = _f(0, minimum=0)
    prompt_tokens: int = _f(0, minimum=0)
    completion_tokens: int = _f(0, minimum=0)
    model_wall_seconds: float = _f(0.0, minimum=0)
    total_wall_seconds: float = _f(0.0, minimum=0)
    cpu_seconds: float | None = _f(None, minimum=0, description="null = not measured (must be in not_run)")
    peak_rss_bytes: int | None = _f(None, minimum=0, description="null = not measured (must be in not_run)")

    MEASURABLE = ("cpu_seconds", "peak_rss_bytes")


@dataclass
class NotRun:
    item: str = _f(minLength=1, maxLength=200)
    reason: str = _f(minLength=1, maxLength=2000)


@dataclass
class ErrorInfo:
    code: str = _f(pattern=r"^[a-z][a-z0-9_]{0,63}$")
    message: str = _f(maxLength=2000)
    retryable: bool = _f()


@dataclass
class ModelInfo:
    id: str = _f(minLength=1, maxLength=500)
    sha256: str | None = _f(None, pattern=r"^[0-9a-f]{64}$")


@dataclass
class EngineResult:
    request_id: str = _f(pattern=ID_PATTERN)
    task: Task = _f()
    status: Literal["ok", "incomplete", "error"] = _f(
        description="ok: an answer was produced (it may still be unverified); incomplete: "
                    "no usable answer within budget; error: see 'error'")
    engine_version: str = _f(minLength=1, maxLength=64)
    contract_version: Literal["1.0"] = "1.0"
    answer: str | None = _f(None, maxLength=200000)
    checked_claims: list[CheckedClaim] = _f(factory=list)
    unverified_claims: list[UnverifiedClaim] = _f(factory=list)
    evidence: list[Evidence] = _f(factory=list)
    executions: list[ExecutionRecord] = _f(factory=list)
    cost: Cost = _f(factory=Cost)
    not_run: list[NotRun] = _f(factory=list)
    model: ModelInfo | None = None
    error: ErrorInfo | None = None

    def problems(self) -> list[str]:
        out = []
        if (self.status == "error") != (self.error is not None):
            out.append("error: required exactly when status is 'error'")
        if self.status == "ok" and self.answer is None:
            out.append("answer: required when status is 'ok'")
        ids = [e.id for e in self.evidence] + [x.id for x in self.executions]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            out.append(f"ids: duplicated across evidence/executions: {duplicates}")
        known = set(ids)
        for n, ev in enumerate(self.evidence):
            if ev.kind == "file_span" and (ev.start_line is None or ev.end_line is None):
                out.append(f"evidence[{n}]: file_span needs start_line and end_line")
            if ev.start_line is not None and ev.end_line is not None and ev.end_line < ev.start_line:
                out.append(f"evidence[{n}]: end_line before start_line")
        for name, claims in (("checked_claims", self.checked_claims),
                             ("unverified_claims", self.unverified_claims)):
            for n, claim in enumerate(claims):
                missing = [i for i in claim.evidence_ids if i not in known]
                if missing:
                    out.append(f"{name}[{n}]: unknown evidence ids {missing}")
        for n, ex in enumerate(self.executions):
            if ex.backend in NON_BOUNDARY_BACKENDS and ex.security_boundary:
                out.append(f"executions[{n}]: backend {ex.backend!r} is not a security boundary")
        listed = {nr.item for nr in self.not_run}
        for name in Cost.MEASURABLE:
            if getattr(self.cost, name) is None and f"cost.{name}" not in listed:
                out.append(f"not_run: unmeasured cost.{name} must be listed")
        return out

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))


def not_measured(cost: Cost, reason: str = "not measured by this route") -> list[NotRun]:
    """NotRun entries for every cost field left as None."""
    return [NotRun(f"cost.{name}", reason) for name in Cost.MEASURABLE if getattr(cost, name) is None]


# --------------------------------------------------------------------------- codec

def _hints(cls) -> dict[str, Any]:
    return get_type_hints(cls)


def _optional_inner(tp):
    if get_origin(tp) in (Union, types.UnionType):
        args = [a for a in get_args(tp) if a is not type(None)]
        if len(args) == 1 and type(None) in get_args(tp):
            return args[0]
    return None


def _check_constraints(value, meta: dict, path: str, problems: list[str]) -> None:
    if isinstance(value, str):
        if "minLength" in meta and len(value) < meta["minLength"]:
            problems.append(f"{path}: shorter than {meta['minLength']}")
        if "maxLength" in meta and len(value) > meta["maxLength"]:
            problems.append(f"{path}: longer than {meta['maxLength']}")
        if "pattern" in meta and not re.search(meta["pattern"], value):
            problems.append(f"{path}: does not match {meta['pattern']}")
    elif isinstance(value, list):
        if "minItems" in meta and len(value) < meta["minItems"]:
            problems.append(f"{path}: needs at least {meta['minItems']} item(s)")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in meta and value < meta["minimum"]:
            problems.append(f"{path}: below {meta['minimum']}")
        if "exclusiveMinimum" in meta and value <= meta["exclusiveMinimum"]:
            problems.append(f"{path}: must be above {meta['exclusiveMinimum']}")
        if "maximum" in meta and value > meta["maximum"]:
            problems.append(f"{path}: above {meta['maximum']}")


def _decode(tp, value, path: str, meta: dict, problems: list[str], strict: bool):
    inner = _optional_inner(tp)
    if inner is not None:
        if value is None:
            return None
        return _decode(inner, value, path, meta, problems, strict)
    origin = get_origin(tp)
    if origin is Literal:
        if not any(type(value) is type(a) and value == a for a in get_args(tp)):
            problems.append(f"{path}: must be one of {list(get_args(tp))}")
        return value
    if origin is list:
        if not isinstance(value, list):
            problems.append(f"{path}: must be an array")
            return []
        _check_constraints(value, meta, path, problems)
        (item_tp,) = get_args(tp)
        return [_decode(item_tp, v, f"{path}[{i}]", {}, problems, strict) for i, v in enumerate(value)]
    if is_dataclass(tp):
        return _from_dict(tp, value, path, problems, strict)
    ok = {bool: type(value) is bool,
          int: type(value) is int,
          float: type(value) in (int, float) and math.isfinite(value),
          str: isinstance(value, str)}.get(tp)
    if ok is None:
        raise TypeError(f"unsupported contract type {tp!r} at {path}")
    if not ok:
        problems.append(f"{path}: must be {tp.__name__}")
        return value
    _check_constraints(value, meta, path, problems)
    return float(value) if tp is float else value


def _from_dict(cls, data, path: str, problems: list[str], strict: bool):
    if not isinstance(data, dict):
        problems.append(f"{path or cls.__name__}: must be an object")
        return None
    hints = _hints(cls)
    names = {f.name for f in fields(cls)}
    if strict:
        for key in sorted(set(data) - names):
            problems.append(f"{path + '.' if path else ''}{key}: unknown field")
    kwargs = {}
    for f in fields(cls):
        where = f"{path}.{f.name}" if path else f.name
        if f.name not in data:
            if f.default is MISSING and f.default_factory is MISSING:
                problems.append(f"{where}: required")
            continue
        kwargs[f.name] = _decode(hints[f.name], data[f.name], where, dict(f.metadata), problems, strict)
    try:
        return cls(**kwargs)
    except TypeError:
        return None  # missing required fields, already reported


def request_from_dict(data: Any) -> EngineRequest:
    """Strictly decode and validate a request; raise ContractError listing every problem."""
    problems: list[str] = []
    request = _from_dict(EngineRequest, data, "", problems, strict=True)
    if request is not None and not problems:
        problems += request.problems()
    if problems:
        raise ContractError(problems)
    return request


def result_from_dict(data: Any, *, strict: bool = True) -> EngineResult:
    """Decode and validate a result; ``strict=False`` ignores unknown (newer-minor) fields."""
    problems: list[str] = []
    if not strict and isinstance(data, dict) and isinstance(data.get("contract_version"), str):
        major = data["contract_version"].split(".", 1)[0]
        if major == CONTRACT_VERSION.split(".", 1)[0]:
            data = {**data, "contract_version": CONTRACT_VERSION}
    result = _from_dict(EngineResult, data, "", problems, strict=strict)
    if result is not None and not problems:
        problems += result.problems()
    if problems:
        raise ContractError(problems)
    return result


def validate_result(result: EngineResult) -> EngineResult:
    """Round-trip a constructed result through the decoder so producers cannot emit
    anything a consumer would reject. Returns the result; raises ContractError."""
    result_from_dict(json.loads(result.to_json()))
    return result


# --------------------------------------------------------------------------- schema

def _schema_for(tp, meta: dict, defs: dict) -> dict:
    inner = _optional_inner(tp)
    if inner is not None:
        return {"anyOf": [_schema_for(inner, meta, defs), {"type": "null"}]}
    origin = get_origin(tp)
    if origin is Literal:
        return {"enum": list(get_args(tp))}
    if origin is list:
        out = {"type": "array", "items": _schema_for(get_args(tp)[0], {}, defs)}
        if "minItems" in meta:
            out["minItems"] = meta["minItems"]
        return out
    if is_dataclass(tp):
        _define(tp, defs)
        return {"$ref": f"#/$defs/{tp.__name__}"}
    out = {"type": {bool: "boolean", int: "integer", float: "number", str: "string"}[tp]}
    for key in ("minLength", "maxLength", "pattern", "minimum", "maximum", "exclusiveMinimum"):
        if key in meta:
            out[key] = meta[key]
    return out


def _define(cls, defs: dict) -> None:
    if cls.__name__ in defs:
        return
    defs[cls.__name__] = {}  # placeholder against recursion
    hints = _hints(cls)
    props, required = {}, []
    for f in fields(cls):
        meta = dict(f.metadata)
        prop = _schema_for(hints[f.name], meta, defs)
        if "description" in meta:
            prop = {**prop, "description": meta["description"]}
        if f.default is not MISSING and not is_dataclass(f.default):
            prop = {**prop, "default": f.default}
        props[f.name] = prop
        if f.default is MISSING and f.default_factory is MISSING:
            required.append(f.name)
    defs[cls.__name__] = {"type": "object", "additionalProperties": False,
                          "properties": props, "required": required}


def json_schema() -> dict:
    """JSON Schema (2020-12) for EngineRequest and EngineResult. Structural rules only:
    cross-field invariants (EngineRequest.problems, EngineResult.problems) are in code."""
    defs: dict = {}
    _define(EngineRequest, defs)
    _define(EngineResult, defs)
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": SCHEMA_ID,
        "title": f"NOEMA Engine contract {CONTRACT_VERSION}",
        "description": "Generated by cpu_platform.contract.json_schema(); do not edit by hand. "
                       "Cross-field rules are documented in docs/engine-contract.md.",
        "oneOf": [{"$ref": "#/$defs/EngineRequest"}, {"$ref": "#/$defs/EngineResult"}],
        "$defs": dict(sorted(defs.items())),
    }


def schema_text() -> str:
    return json.dumps(json_schema(), indent=2, ensure_ascii=False) + "\n"


# --------------------------------------------------------------------------- v1 adapters

def _v1_cost(usage: dict) -> Cost:
    return Cost(model_calls=int(usage.get("model_calls", 0)),
                prompt_tokens=int(usage.get("prompt_tokens") or 0),
                completion_tokens=int(usage.get("completion_tokens") or 0),
                model_wall_seconds=round(float(usage.get("total_model_latency_ms", 0)) / 1000, 6),
                total_wall_seconds=round(float(usage.get("total_latency_ms", 0)) / 1000, 6))


def from_v1_ask(question: str, legacy: dict, *, request_id: str, engine_version: str,
                model_id: str | None = None) -> EngineResult:
    """Map a /v1/ask result onto the contract without changing what it claims.

    Exact arithmetic becomes one verified claim. A model answer becomes an unverified
    claim; each arithmetic equality the audit recomputed becomes a checked claim about
    that equation only, never about the answer.
    """
    cost = _v1_cost(legacy.get("usage", {}))
    evidence = [Evidence(id="question", kind="input", source="question", excerpt=question[:MAX_EXCERPT_CHARS])]
    checked, unverified, not_run = [], [], []
    answer = legacy.get("answer")
    status = {"ok": "ok", "incomplete": "incomplete"}.get(legacy.get("status"), "error")
    error = None
    if legacy.get("route") == "exact_arithmetic":
        if status == "ok":
            checked.append(CheckedClaim(
                claim=f"The arithmetic expression in the question evaluates to {answer}.",
                result="verified", checker="exact_arithmetic",
                scope=legacy.get("verification_scope") or "exact evaluation of the numeric expression",
                evidence_ids=["question"]))
        else:
            error = ErrorInfo("invalid_expression", str(legacy.get("error") or "invalid expression"), False)
    else:
        if answer is not None:
            evidence.append(Evidence(id="answer", kind="model_output", source="model",
                                     excerpt=answer[:MAX_EXCERPT_CHARS], retrieved_by=legacy.get("route")))
            unverified.append(UnverifiedClaim(
                claim=answer, reason=legacy.get("verification_scope") or "model output; no checker ran",
                evidence_ids=["answer"]))
        checks = legacy.get("checks") or {}
        arithmetic = checks.get("arithmetic") or {}
        for n, eq in enumerate(arithmetic.get("equations") or []):
            if eq.get("correct") is None or answer is None:
                continue
            notes = [f"operand not found in the question: {op}" for op in eq.get("ungrounded") or ()]
            if eq.get("rounded"):
                notes.append("accepted as the stated rounding of the exact result")
            if not eq["correct"] and eq.get("actual") is not None:
                notes.append(f"exact value is {eq['actual']}")
            checked.append(CheckedClaim(
                claim=f"{eq['expression']} = {eq['claimed']}",
                result="verified" if eq["correct"] else "refuted",
                checker="cot_check.arithmetic",
                scope="this equality was recomputed exactly; it does not show the equation is "
                      "relevant to the question or that the answer is correct",
                evidence_ids=["answer"], notes=notes))
        if arithmetic.get("truncated"):
            not_run.append(NotRun("arithmetic audit coverage", "text truncated or equation limit reached"))
        for issue in checks.get("issues") or []:
            not_run.append(NotRun("issue", issue))
        if legacy.get("domain") == "code":
            not_run.append(NotRun("code execution", "v1 /ask checks Python syntax only; code was not run"))
        if legacy.get("domain") == "factual":
            not_run.append(NotRun("source retrieval", "no external or repository sources were retrieved"))
    not_run += not_measured(cost)
    result = EngineResult(
        request_id=request_id, task="ask", status=status, engine_version=engine_version,
        answer=answer if status == "ok" else None, checked_claims=checked,
        unverified_claims=unverified, evidence=evidence, cost=cost, not_run=not_run,
        model=ModelInfo(model_id) if model_id else None, error=error)
    return validate_result(result)


def from_v1_logic(premises: list[str], conclusion: str, legacy: dict, *, request_id: str,
                  engine_version: str) -> EngineResult:
    """Map a /v1/verify-logic result: one checked claim, no model involved."""
    cost = _v1_cost(legacy.get("usage", {}))
    text = "Premises: " + "; ".join(premises) + " | Conclusion: " + conclusion
    entailed = legacy.get("answer") == "entailed"
    result = EngineResult(
        request_id=request_id, task="ask", status="ok", engine_version=engine_version,
        answer=legacy.get("answer"),
        checked_claims=[CheckedClaim(
            claim=f"The premises {'entail' if entailed else 'do not entail'} the conclusion.",
            result="verified", checker="propositional_logic",
            scope=str(legacy.get("verification_scope") or "classical propositional entailment"),
            evidence_ids=["input"])],
        evidence=[Evidence(id="input", kind="input", source="question", excerpt=text[:MAX_EXCERPT_CHARS])],
        cost=cost, not_run=not_measured(cost))
    return validate_result(result)


def main(argv: list[str] | None = None) -> int:
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description="Write or check the generated contract JSON Schema")
    parser.add_argument("--write", metavar="PATH", help="write the schema to PATH")
    parser.add_argument("--check", metavar="PATH", help="exit 1 if PATH differs from the generated schema")
    args = parser.parse_args(argv)
    if args.write:
        Path(args.write).write_text(schema_text(), encoding="utf-8")
        return 0
    if args.check:
        same = Path(args.check).read_text(encoding="utf-8") == schema_text()
        print("schema up to date" if same else f"{args.check} is stale: regenerate with --write")
        return 0 if same else 1
    print(schema_text(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
