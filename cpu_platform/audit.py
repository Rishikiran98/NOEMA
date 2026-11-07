"""Audit reports (roadmap M3.6): one Markdown and one JSON report per request, built from the
request and its ``EngineResult``. Standard library only.

    audit = build(request, result)              # dict (JSON-safe)
    text = to_markdown(audit)
    write(request, result, "audits/")           # <request_id>.audit.json and .audit.md

    python -m cpu_platform.audit render RESULT.json --request REQUEST.json --out audits/
    python -m cpu_platform.audit digest PATH                     # repository fingerprint
    python -m cpu_platform.audit span PATH FILE START END [--quote TEXT --kind inline|block]
    python -m cpu_platform.audit save REPORT.json EVIDENCE_ID [--out FILE]

The report covers the question, the repository state, retrieved evidence, citations and
their status, commands run, test results, the patch, cost and what was NOT checked. Every
verified or refuted claim comes with how to reproduce it from the report alone: the steps, the
commands (with ``{repo}`` standing for the repository and ``{file:ID}`` for a file saved from
evidence ``ID``), and the expected result. A claim the report cannot support that way says so
(``reproducible: false`` and why), and is listed under "NOT checked".

What reproducing shows is only what the checker showed: a passing test run does not make a
fix correct, and an existing quote does not make a citation relevant.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime, timezone

from .citation_check import safe_relative
from .contract import MAX_EXCERPT_CHARS, EngineRequest, EngineResult, Evidence, request_from_dict, result_from_dict

AUDIT_VERSION = "1.0"
REPO_EVIDENCE_ID = "repo"
BACKEND_ARGS = ["--backend", "container"]
BACKEND_NOTE = ("`--backend container` needs Docker; without it use `--backend dev "
                "--i-understand-dev-is-not-isolated`, which is NOT a security boundary (the patch is model output)")
DIGEST_DEFINITION = ("SHA-256 over the sorted lines `<path>\\0<sha256 of file bytes>\\n` for every file a NOEMA "
                     "sandbox copies (.gitignore applied; secret-named, symlinked and oversized files left out)")
SPAN_DEFINITION = ("SHA-256 of lines START..END decoded as UTF-8 and joined with '\\n' "
                   "(line endings normalized, no trailing newline)")
FAILING = ("failed", "error")


# --------------------------------------------------------------------------- repository state

def tree_digest(path: str | Path) -> tuple[str, int]:
    from .sandbox import copy_selection
    root = Path(path)
    h = hashlib.sha256()
    count = 0
    for rel in sorted(copy_selection(root)):
        h.update(f"{rel}\0{hashlib.sha256((root / rel).read_bytes()).hexdigest()}\n".encode("utf-8"))
        count += 1
    return h.hexdigest(), count


def git_state(path: str | Path) -> dict | None:
    """HEAD commit and whether the working tree differs from it, or None (no git, not a repo)."""
    env = {k: v for k, v in os.environ.items() if k in ("PATH", "SYSTEMROOT", "HOME", "USERPROFILE", "LANG")}
    env["GIT_OPTIONAL_LOCKS"] = "0"
    try:
        top = subprocess.run(["git", "-C", str(path), "rev-parse", "--show-toplevel", "HEAD"], capture_output=True,
                             text=True, timeout=15, env=env)
        if top.returncode != 0:
            return None
        toplevel, commit = top.stdout.split()[:2]
        status = subprocess.run(["git", "-C", str(path), "status", "--porcelain", "--", "."], capture_output=True,
                                text=True, timeout=30, env=env)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    return {"commit": commit, "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
            "subdirectory": Path(path).resolve() != Path(toplevel).resolve()}


def repository_state(path: str | Path) -> dict:
    digest, files = tree_digest(path)
    return {"tree_sha256": digest, "files": files, "git": git_state(path), "definition": DIGEST_DEFINITION}


def repository_evidence(path: str | Path) -> Evidence:
    """The repository's state when the request ran, as evidence ``repo`` (taken by the routes)."""
    state = repository_state(path)
    return Evidence(id=REPO_EVIDENCE_ID, kind="input", source="repository", sha256=state["tree_sha256"],
                    excerpt=json.dumps({k: state[k] for k in ("files", "git")}), retrieved_by="audit.tree_digest")


def test_summary(run, select, patch_id) -> str:
    """JSON for a test run's evidence: selection, patch, counts and every non-passing test.
    Passing tests are counted, not listed, so the summary fits the excerpt limit."""
    not_passing = {t.id: t.outcome for t in run.tests if t.outcome != "passed"}
    out = {"framework": run.framework, "status": run.status, "select": list(select) if select else None,
           "patch": patch_id, "counts": run.counts, "not_passing": not_passing, "truncated": False}
    text = json.dumps(out)
    while len(text) > MAX_EXCERPT_CHARS and out["not_passing"]:
        out["not_passing"] = dict(list(out["not_passing"].items())[:len(out["not_passing"]) // 2])
        out["truncated"] = True
        text = json.dumps(out)
    if out["select"] and len(text) > MAX_EXCERPT_CHARS:
        out["select"], out["truncated"] = out["select"][:20], True
        text = json.dumps(out)
    return text[:MAX_EXCERPT_CHARS]


# --------------------------------------------------------------------------- reproduction recipes

def _json(text: str | None) -> dict | None:
    try:
        value = json.loads(text or "")
        return value if isinstance(value, dict) else None
    except ValueError:
        return None


def _no(reason: str) -> dict:
    return {"reproducible": False, "why_not": reason, "steps": [], "commands": [], "files": {}}


def _maybe_truncated(ev: Evidence) -> bool:
    return ev.excerpt is None or len(ev.excerpt) >= MAX_EXCERPT_CHARS


def _patch_file(patch_id: str | None, ev: dict, result: EngineResult) -> tuple[dict | None, str | None]:
    """{"D1.diff": {...}} for the applied patch, or (None, reason)."""
    patch = ev.get(patch_id)
    if patch is None:
        return None, f"patch evidence {patch_id} is missing"
    text = patch.excerpt or ""
    if _maybe_truncated(patch):
        full = result.answer or ""
        if patch.sha256 and hashlib.sha256(full.encode("utf-8")).hexdigest() == patch.sha256:
            return {f"{patch_id}.diff": {"source": "answer", "sha256": patch.sha256}}, None
        return None, f"patch {patch_id} is longer than the report keeps ({MAX_EXCERPT_CHARS} characters)"
    return {f"{patch_id}.diff": {"source": patch_id, "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}}, None


def _plain_test_command(framework: str, select) -> str:
    if framework == "pytest":
        return "python -m pytest " + " ".join(select or [])
    if select:
        return "python -m unittest " + " ".join(select)
    return "python -m unittest discover -s tests"


def _testrun_recipe(claim, ev: dict, result: EngineResult) -> dict:
    summaries = [ev[i] for i in claim.evidence_ids if i in ev and ev[i].kind == "command_output"]
    summary = _json(summaries[0].excerpt) if summaries else None
    if summary is None:
        return _no("the test run's selection and outcomes are not in the result")
    if summary.get("truncated"):
        return _no("the test run's selection or outcomes were too long to keep in the report")
    select = summary.get("select") or []
    verified = claim.result == "verified"
    files, steps = {}, ["Start from the repository at the state given under 'Repository state' "
                        "(check it with `python -m cpu_platform.audit digest {repo}`)."]
    if summary.get("patch"):
        files, why = _patch_file(summary["patch"], ev, result)
        if files is None:
            return _no(why)
        name = next(iter(files))
        steps.append(f"Save the patch {summary['patch']} as `{name}` (`python -m cpu_platform.audit save "
                     f"REPORT.json {summary['patch']} --out {name}`).")
        argv = ["python", "-m", "cpu_platform.patch", "try", f"{{file:{name}}}", "--repo", "{repo}", *BACKEND_ARGS]
        if select:
            argv += ["--select", *select]
        expect = [{"path": "applied", "op": "equals", "value": True},
                  {"path": "tests.status", "op": "equals" if verified else "not_equals", "value": "passed"}]
        steps.append("Apply it in a sandbox copy and run the tests; the original repository is not changed.")
        plain = (f"Independently, in a disposable copy: `git apply {name}` then `"
                 f"{_plain_test_command(summary['framework'], select)}`.")
    else:
        argv = ["python", "-m", "cpu_platform.testrun", "{repo}", *BACKEND_ARGS]
        if select:
            argv += ["--select", *select]
        failing = sorted(i for i, o in summary["not_passing"].items() if o in FAILING)
        expect = [{"path": "tests", "op": "failing_ids_equal", "value": failing}]
        steps.append("Run the tests in a sandbox copy without any patch.")
        plain = f"Independently, in a disposable copy: `{_plain_test_command(summary['framework'], select)}`."
    if summary["framework"] != "pytest" and not select:
        plain += " Without a tests/ folder: `python -m unittest discover`."
    steps.append(plain + " Results come from the code under test, so they could be forged by malicious code; "
                 "flaky tests can differ between runs.")
    return {"reproducible": True, "steps": steps, "commands": [{"argv": argv, "expect": expect}], "files": files,
            "notes": [BACKEND_NOTE]}


def _answer_file(claim, ev: dict) -> tuple[dict | None, str | None, str | None]:
    pid = next((i for i in claim.evidence_ids if re.fullmatch(r"P\d+", i)), None)
    if pid is None or pid not in ev:
        return None, None, "the model answer this check read is not in the result"
    if _maybe_truncated(ev[pid]):
        return None, None, f"the model answer {pid} is longer than the report keeps ({MAX_EXCERPT_CHARS} characters)"
    name = f"{pid}.txt"
    return {name: {"source": pid, "sha256": hashlib.sha256(ev[pid].excerpt.encode("utf-8")).hexdigest()}}, name, None


def _patch_recipe(claim, ev: dict, apply_check: bool) -> dict:
    files, name, why = _answer_file(claim, ev)
    if files is None:
        return _no(why)
    verified = claim.result == "verified"
    save = f"Save the model's answer {name[:-4]} as `{name}` (`python -m cpu_platform.audit save REPORT.json {name[:-4]} --out {name}`)."
    if apply_check:
        argv = ["python", "-m", "cpu_platform.patch", "try", f"{{file:{name}}}", "--repo", "{repo}", *BACKEND_ARGS,
                "--no-tests"]
        expect = [{"path": "applied", "op": "equals", "value": verified}]
        what = "Apply the diff in a sandbox copy (no tests); lenient reading is reported under `problems`."
    else:
        argv = ["python", "-m", "cpu_platform.patch", "check", f"{{file:{name}}}", "--repo", "{repo}"]
        expect = [{"path": "ok", "op": "equals", "value": verified}]
        what = "Check the diff against the patch policy (nothing is applied)."
    return {"reproducible": True, "steps": [save, what], "commands": [{"argv": argv, "expect": expect}],
            "files": files, "notes": [BACKEND_NOTE] if apply_check else []}


CITE_FILE = re.compile(r"^cited file (?P<path>.+) exists$")
CITE_LINES = re.compile(r"^lines (?P<s>\d+)-(?P<e>\d+) exist in (?P<path>.+)$")
CITE_QUOTE = re.compile(r"^`(?P<q>.*)` appears in (?P<path>.+):(?P<s>\d+)-(?P<e>\d+)$", re.S)


def _citation_recipe(claim, ev: dict) -> dict:
    verified = claim.result == "verified"
    span = next((ev[i] for i in claim.evidence_ids if i in ev and ev[i].kind == "file_span"), None)
    if claim.checker == "citation_check.file" and (m := CITE_FILE.match(claim.claim)):
        argv = ["python", "-m", "cpu_platform.audit", "span", "{repo}", m["path"], "1", "1"]
        expect = [{"path": "exists", "op": "equals", "value": verified}]
        steps = [f"Check that `{m['path']}` exists in the repository (a path that is absolute, escapes the "
                 "repository or passes through a symlink counts as not existing)."]
    elif claim.checker == "citation_check.lines" and (m := CITE_LINES.match(claim.claim)):
        argv = ["python", "-m", "cpu_platform.audit", "span", "{repo}", m["path"], m["s"], m["e"]]
        expect = [{"path": "in_range", "op": "equals", "value": verified}]
        if verified and span is not None and span.sha256 and (span.start_line, span.end_line) == (int(m["s"]), int(m["e"])):
            expect.append({"path": "sha256", "op": "equals", "value": span.sha256})
        steps = [f"Open `{m['path']}` and check that lines {m['s']}-{m['e']} exist"
                 + (f"; their {SPAN_DEFINITION} must be the one shown." if len(expect) > 1 else ".")]
    elif claim.checker == "citation_check.quote" and (m := CITE_QUOTE.match(claim.claim)):
        quote = m["q"]
        if len(quote) >= 300:
            return _no("the quote is longer than the claim keeps (300 characters)")
        kind = "block" if any("block" in n for n in claim.notes) else "inline"
        argv = ["python", "-m", "cpu_platform.audit", "span", "{repo}", m["path"], m["s"], m["e"],
                "--quote", quote, "--kind", kind]
        expect = [{"path": "quote_found", "op": "equals", "value": verified}]
        steps = [f"Open `{m['path']}` lines {m['s']}-{m['e']} and look for the quote "
                 + ("as a substring." if kind == "inline" else
                    "as consecutive lines (trailing whitespace and common indentation ignored).")]
    else:
        return _no(f"the claim text does not have the form checker {claim.checker} writes")
    return {"reproducible": True, "steps": steps + ["This shows the span exists and contains the text; whether it "
                                                    "supports the answer is NOT CHECKED."],
            "commands": [{"argv": argv, "expect": expect}], "files": {}, "notes": []}


def _manual(steps: list[str]) -> dict:
    return {"reproducible": True, "steps": steps, "commands": [], "files": {}, "notes": ["manual check"]}


def recipe(claim, ev: dict, result: EngineResult) -> dict:
    checker = claim.checker
    if checker.startswith("testrun."):
        return _testrun_recipe(claim, ev, result)
    if checker in ("patch.validate", "patch.apply"):
        return _patch_recipe(claim, ev, checker == "patch.apply")
    if checker.startswith("citation_check."):
        return _citation_recipe(claim, ev)
    if checker == "injection.canary" and (m := re.search(r"canary (\S+)$", claim.claim)):
        return _manual([f"Search the answer above for `{m[1]}`, ignoring case: it must be "
                        + ("absent." if claim.result == "verified" else "present.")])
    if checker == "injection.prompt_leak":
        from .injection_eval import LEAK_PROBES
        return _manual(["Search the answer above, ignoring case, for each of: "
                        + "; ".join(f"`{p}`" for p in LEAK_PROBES) + ". "
                        + ("None may appear." if claim.result == "verified" else "At least one appears.")])
    if checker == "exact_arithmetic":
        return _manual(["Evaluate the arithmetic expression in the question exactly (rational arithmetic, "
                        "for example Python's `fractions.Fraction`); it must equal the answer."])
    if checker == "cot_check.arithmetic" and " = " in claim.claim:
        expression, claimed = claim.claim.rsplit(" = ", 1)
        return _manual([f"Evaluate `{expression}` exactly; it must "
                        + ("equal" if claim.result == "verified" else "differ from") + f" `{claimed}`"
                        + (" (allowing the stated rounding)" if any("rounding" in n for n in claim.notes) else "") + "."])
    if checker == "propositional_logic":
        return _manual(["Build the truth table of the premises and conclusion in the input evidence; the claim "
                        "holds if every row that makes all premises true also makes the conclusion true "
                        "(or, for 'do not entail', some such row makes it false)."])
    return _no(f"no reproduction recipe for checker {checker!r}")


def check_expectations(output: dict, expect: list[dict]) -> list[str]:
    """Problems found comparing a command's JSON output with its expectations ([] = as expected).
    The same comparisons a reader makes by eye; kept here so reports can be tested."""
    problems = []
    for e in expect:
        value = output
        for key in e["path"].split("."):
            value = value.get(key) if isinstance(value, dict) else None
        if e["op"] == "equals" and value != e["value"]:
            problems.append(f"{e['path']}: expected {e['value']!r}, got {value!r}")
        elif e["op"] == "not_equals" and value == e["value"]:
            problems.append(f"{e['path']}: expected anything but {e['value']!r}")
        elif e["op"] == "failing_ids_equal":
            got = sorted(t["id"] for t in (value or []) if t.get("outcome") in FAILING)
            if got != sorted(e["value"]):
                problems.append(f"failing tests: expected {e['value']}, got {got}")
    return problems


def describe(e: dict) -> str:
    if e["op"] == "failing_ids_equal":
        return "exactly these tests fail or error: " + (", ".join(f"`{i}`" for i in e["value"]) or "none")
    return f"`{e['path']}` {'is' if e['op'] == 'equals' else 'is not'} `{json.dumps(e['value'])}`"


# --------------------------------------------------------------------------- the report

def build(request: EngineRequest, result: EngineResult, *, generated_at: str | None = None) -> dict:
    ev = {e.id: e for e in result.evidence}
    claims, extra_not_checked = [], []
    for n, claim in enumerate(result.checked_claims, 1):
        how = recipe(claim, ev, result)
        claims.append({"id": f"K{n}", "claim": claim.claim, "result": claim.result, "checker": claim.checker,
                       "scope": claim.scope, "notes": claim.notes, "evidence_ids": claim.evidence_ids,
                       "reproduce": how})
        if not how["reproducible"]:
            extra_not_checked.append({"item": f"reproduction of K{n}", "reason": how["why_not"]})
    repo = None
    if REPO_EVIDENCE_ID in ev and ev[REPO_EVIDENCE_ID].retrieved_by == "audit.tree_digest":
        info = _json(ev[REPO_EVIDENCE_ID].excerpt) or {}
        repo = {"tree_sha256": ev[REPO_EVIDENCE_ID].sha256, "files": info.get("files"), "git": info.get("git"),
                "definition": DIGEST_DEFINITION}
    elif request.repo is not None:
        extra_not_checked.append({"item": "repository state",
                                  "reason": "the result does not record the repository's state; claims about files "
                                            "and tests cannot be tied to a version of it"})
    tests = []
    for e in result.evidence:
        if e.kind == "command_output" and e.retrieved_by and e.retrieved_by.startswith("testrun."):
            summary = _json(e.excerpt)
            if summary is not None:
                tests.append({"execution": e.source, **summary})
    citations = [{"citation": u.claim[len("citation "):], "status": u.reason}
                 for u in result.unverified_claims if u.claim.startswith("citation ")]
    req = request.__dict__
    return {
        "audit_version": AUDIT_VERSION,
        "generated_at": generated_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "request": {"question": request.question, "task": request.task,
                    "repo": None if request.repo is None else {"path": request.repo.path, "revision": request.repo.revision},
                    "allow_execution": request.allow_execution, "budget": request.budget.__dict__,
                    "thinking": req.get("thinking"), "client_request_id": request.client_request_id},
        "result": {"request_id": result.request_id, "task": result.task, "status": result.status,
                   "engine_version": result.engine_version, "contract_version": result.contract_version,
                   "model": None if result.model is None else result.model.__dict__,
                   "error": None if result.error is None else result.error.__dict__},
        "repository": repo,
        "answer": result.answer,
        "patch": result.answer if result.task == "repo_fix" else None,
        "claims": claims,
        "unverified_claims": [u.__dict__ for u in result.unverified_claims],
        "citations": citations,
        "evidence": [e.__dict__ for e in result.evidence],
        "executions": [x.__dict__ for x in result.executions],
        "tests": tests,
        "cost": result.cost.__dict__,
        "not_checked": [n.__dict__ for n in result.not_run] + extra_not_checked,
    }


def _fence(text: str, lang: str = "") -> str:
    longest = max((len(m) for m in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}{lang}\n{text.rstrip(chr(10))}\n{fence}"


def _cell(text) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _shell(argv: list[str]) -> str:
    def quote(arg: str) -> str:
        arg = arg.replace("{repo}", "<REPO>")
        arg = re.sub(r"\{file:([^}]+)\}", r"\1", arg)
        return arg if re.fullmatch(r"[\w./:<>=@+-]+", arg) else '"' + arg.replace('"', '\\"') + '"'
    return " ".join(quote(a) for a in argv)


def to_markdown(audit: dict) -> str:
    r, q = audit["result"], audit["request"]
    out = [f"# NOEMA audit report: {r['request_id']}", "",
           f"Status **{r['status']}**, task `{r['task']}`, engine {r['engine_version']}, contract "
           f"{r['contract_version']}, model {('`' + r['model']['id'] + '`') if r['model'] else 'none'}, "
           f"generated {audit['generated_at']}.", ""]
    if r["error"]:
        out += [f"**Error** `{r['error']['code']}`: {r['error']['message']}", ""]
    out += ["Verified means a named checker checked that specific claim; it says nothing beyond the "
            "checker's scope. Passing tests do not make a fix correct.", "", "## Request", "",
            _fence(q["question"]), "",
            f"- task: `{q['task']}`; execution allowed: {q['allow_execution']}",
            f"- repository: {('`' + q['repo']['path'] + '`' + (' at ' + q['repo']['revision'] if q['repo']['revision'] else ' (working tree)')) if q['repo'] else 'none'}",
            f"- budget: {q['budget']['max_model_calls']} model call(s), {q['budget']['max_output_tokens']} output "
            f"tokens, {q['budget']['max_wall_seconds']} s", ""]
    out += ["## Repository state", ""]
    repo = audit["repository"]
    if repo:
        git = repo["git"]
        out += [f"- tree SHA-256: `{repo['tree_sha256']}` ({repo['files']} files)",
                f"- git: {'commit `' + git['commit'] + '`' + (', working tree modified' if git['dirty'] else ', clean') if git else 'not a git repository (or git unavailable)'}",
                f"- definition: {repo['definition']}",
                "- check: `python -m cpu_platform.audit digest <REPO>` must print the same `tree_sha256`. "
                "Below, `<REPO>` is the repository in that state. On Windows, `py` instead of `python`.", ""]
    else:
        out += ["NOT RECORDED: claims cannot be tied to a version of the repository.", ""]
    out += ["## Answer", "", _fence(audit["answer"]) if audit["answer"] is not None else "No answer.", ""]
    claims = audit["claims"]
    out += ["## Checked claims", ""]
    if claims:
        out += ["| # | Result | Claim | Checker | Reproducible |", "|---|---|---|---|---|"]
        out += [f"| {c['id']} | {c['result'].upper()} | {_cell(c['claim'][:200])} | `{c['checker']}` | "
                f"{'yes' if c['reproduce']['reproducible'] else 'NO'} |" for c in claims]
        out.append("")
    else:
        out += ["None: nothing in this result was checked.", ""]
    for c in claims:
        how = c["reproduce"]
        out += [f"### {c['id']}: {c['result'].upper()}", "", _fence(c["claim"]), "",
                f"- checker `{c['checker']}`; scope: {c['scope']}",
                f"- evidence: {', '.join('`' + i + '`' for i in c['evidence_ids'])}"]
        out += [f"- note: {n}" for n in c["notes"]]
        if not how["reproducible"]:
            out += [f"- **reproduction: NOT POSSIBLE from this report**: {how['why_not']}", ""]
            continue
        out += ["", "To reproduce:", ""]
        out += [f"{k}. {s.replace('{repo}', '<REPO>')}" for k, s in enumerate(how["steps"], 1)]
        for cmd in how["commands"]:
            out += ["", _fence(_shell(cmd["argv"]), "bash"), "",
                    "Expected (JSON output): " + "; ".join(describe(e) for e in cmd["expect"]) + "."]
        out += [f"\n_{n}_" for n in how.get("notes", [])]
        out.append("")
    if audit["patch"]:
        out += ["## Patch", "", "The change that was applied and tested (regenerated from the files).", "",
                _fence(audit["patch"], "diff"), ""]
    if audit["citations"]:
        out += ["## Citations", "", "| Citation | Status |", "|---|---|"]
        out += [f"| `{_cell(c['citation'])}` | {_cell(c['status'])} |" for c in audit["citations"]]
        out.append("")
    out += ["## Unverified claims", ""]
    out += [f"- {_cell(u['claim'][:500])}: {u['reason']}" for u in audit["unverified_claims"]] or ["None."]
    out += ["", "## Test results", ""]
    if audit["tests"]:
        out += ["| Execution | Framework | Patch | Status | Counts | Not passing |", "|---|---|---|---|---|---|"]
        for t in audit["tests"]:
            counts = ", ".join(f"{k} {v}" for k, v in t["counts"].items() if v)
            failing = ", ".join(f"`{i}` {o}" for i, o in t["not_passing"].items()) or "none"
            out.append(f"| `{t['execution']}` | {t['framework']} | {t['patch'] or 'none'} | {t['status']} | "
                       f"{counts} | {_cell(failing)}{' (truncated)' if t['truncated'] else ''} |")
    else:
        out.append("No tests were run.")
    out += ["", "## Commands run", ""]
    if audit["executions"]:
        out += ["| Id | Backend | Security boundary | Network | Exit | Timed out | Seconds |", "|---|---|---|---|---|---|---|"]
        out += [f"| `{x['id']}` | {x['backend']} | {'yes' if x['security_boundary'] else '**NO**'} | "
                f"{'yes' if x['network'] else 'no'} | {x['exit_code']} | {x['timed_out']} | {x['duration_s']:.1f} |"
                for x in audit["executions"]]
        for x in audit["executions"]:
            out += ["", f"`{x['id']}`: " + _fence(_shell(x["command"]))]
            for name in ("stdout_tail", "stderr_tail"):
                if x[name].strip():
                    out += ["", f"{name} (last {min(len(x[name]), 2000)} characters"
                            + (", output truncated" if x["output_truncated"] else "") + "):", "",
                            _fence(x[name][-2000:])]
    else:
        out.append("No commands were run.")
    out += ["", "## Evidence", ""]
    for e in audit["evidence"]:
        where = e["source"] + (f":{e['start_line']}-{e['end_line']}" if e["start_line"] else "")
        out += [f"### `{e['id']}` ({e['kind']}): {where}", ""]
        meta = [f"retrieved by `{e['retrieved_by']}`" if e["retrieved_by"] else "",
                f"sha256 `{e['sha256']}`" if e["sha256"] else ""]
        if any(meta):
            out += ["; ".join(m for m in meta if m), ""]
        if e["excerpt"]:
            out += [_fence(e["excerpt"]), ""]
            if len(e["excerpt"]) >= MAX_EXCERPT_CHARS:
                out += [f"(excerpt cut at {MAX_EXCERPT_CHARS} characters)", ""]
    c = audit["cost"]
    out += ["## Cost", "",
            f"{c['model_calls']} model call(s), {c['prompt_tokens']} prompt + {c['completion_tokens']} completion "
            f"tokens, {c['model_wall_seconds']:.1f} s model time, {c['total_wall_seconds']:.1f} s total; CPU seconds "
            f"{c['cpu_seconds'] if c['cpu_seconds'] is not None else 'NOT MEASURED'}, peak RSS "
            f"{c['peak_rss_bytes'] if c['peak_rss_bytes'] is not None else 'NOT MEASURED'}.", "",
            "## NOT checked", ""]
    out += [f"- **{n['item']}**: {n['reason']}" for n in audit["not_checked"]] or ["Nothing listed."]
    return "\n".join(out) + "\n"


def write(request: EngineRequest, result: EngineResult, directory: str | Path) -> tuple[Path, Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    audit = build(request, result)
    stem = re.sub(r"[^A-Za-z0-9_.-]", "_", result.request_id)
    json_path, md_path = directory / f"{stem}.audit.json", directory / f"{stem}.audit.md"
    json_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(to_markdown(audit), encoding="utf-8")
    return json_path, md_path


def evidence_text(audit: dict, evidence_id: str) -> str:
    """The text a recipe's ``{file:...}`` refers to: an evidence excerpt, or the full answer."""
    if evidence_id == "answer" and audit.get("answer") is not None:
        return audit["answer"]
    for e in audit["evidence"]:
        if e["id"] == evidence_id and e["excerpt"] is not None:
            if e["sha256"] and hashlib.sha256(e["excerpt"].encode("utf-8")).hexdigest() != e["sha256"] \
                    and audit.get("answer") and hashlib.sha256(audit["answer"].encode("utf-8")).hexdigest() == e["sha256"]:
                return audit["answer"]           # the excerpt was cut; the full text is the answer
            return e["excerpt"]
    raise KeyError(f"no evidence {evidence_id!r} with text in this report")


# --------------------------------------------------------------------------- span check

def span(root: str | Path, path: str, start: int, end: int, quote: str | None = None, kind: str = "inline") -> dict:
    from .citation_check import quote_found
    root = Path(root).resolve()
    rel = safe_relative(path)
    out = {"path": path, "exists": False, "line_count": None, "in_range": False, "sha256": None, "lines": None}
    if rel is not None and rel == path:
        target = root / rel
        parts, current, through_link = rel.split("/"), root, False
        for part in parts:
            current = current / part
            through_link = through_link or current.is_symlink()
        if not through_link and target.is_file() and target.resolve().is_relative_to(root):
            lines = target.read_bytes().decode("utf-8-sig", errors="replace").splitlines()
            out.update(exists=True, line_count=len(lines), in_range=1 <= start <= end <= len(lines))
            if out["in_range"]:
                chosen = lines[start - 1:end]
                out["sha256"] = hashlib.sha256("\n".join(chosen).encode("utf-8")).hexdigest()
                out["lines"] = chosen
    if quote is not None:
        out["quote_found"] = bool(out["lines"] is not None and quote_found(kind, quote, out["lines"]))
    return out


# --------------------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="NOEMA audit reports and the checks they ask a reader to repeat")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("render", help="write <request_id>.audit.json and .audit.md")
    p.add_argument("result")
    p.add_argument("--request", required=True)
    p.add_argument("--out", default=".")
    p = sub.add_parser("digest", help="repository fingerprint, as recorded in reports")
    p.add_argument("path")
    p = sub.add_parser("span", help="check a file span (and optionally a quote) for a citation claim")
    p.add_argument("path")
    p.add_argument("file")
    p.add_argument("start", type=int)
    p.add_argument("end", type=int)
    p.add_argument("--quote")
    p.add_argument("--kind", choices=("inline", "block"), default="inline")
    p = sub.add_parser("save", help="write an evidence excerpt from a report (a patch, an answer) to a file")
    p.add_argument("report")
    p.add_argument("evidence_id")
    p.add_argument("--out")
    args = parser.parse_args(argv)
    if args.command == "render":
        request = request_from_dict(json.loads(Path(args.request).read_text(encoding="utf-8")))
        result = result_from_dict(json.loads(Path(args.result).read_text(encoding="utf-8")))
        for path in write(request, result, args.out):
            print(path)
        return 0
    if args.command == "digest":
        print(json.dumps(repository_state(args.path), indent=2))
        return 0
    if args.command == "span":
        print(json.dumps(span(args.path, args.file, args.start, args.end, args.quote, args.kind), indent=2))
        return 0
    audit = json.loads(Path(args.report).read_text(encoding="utf-8"))
    try:
        text = evidence_text(audit, args.evidence_id)
    except KeyError as exc:
        parser.error(str(exc))
    if args.out:
        Path(args.out).write_bytes(text.encode("utf-8"))
        print(args.out)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
