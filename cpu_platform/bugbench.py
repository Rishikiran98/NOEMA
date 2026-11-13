"""Injected-bug benchmark (roadmap M4.1). Standard library only. Uses ``cpu_platform.datasets``
for the eval-run log and the same freeze rules (seeded split, pinned SHA-256, eval guarded).

    python -m cpu_platform.bugbench build --noema-commit COMMIT     # generate, test, keep, freeze
    python -m cpu_platform.bugbench check                           # every mutant still applies
    python -m cpu_platform.bugbench run --split dev --limit 20 --backend container \
        --expect-model qwen3-coder-30b-a3b -n 4 --out bugbench_dev.jsonl
    python -m cpu_platform.bugbench status

A mutant is one small bug from ``cpu_platform.mutate`` in a function of a pristine base
repository. **Only mutants the tests catch are kept**: the test modules that mention the
function pass on the base and fail with the mutant. The task is then the M3.5 task: restore
passing tests without editing tests. The reference fix (the inverse edit) is stored in the
benchmark but never shown to the engine.

Base repositories:
* ``noema``: NOEMA itself at a pinned commit, materialized with ``git archive``. Modules that
  manage processes, sandboxes or the network are not mutated (their mutants could make tests act
  on the host during the build, which uses the dev backend).
* ``more_itertools``: more-itertools 10.5.0 (MIT), vendored unchanged in ``data/bugbench/repos``.

Splits: dev and eval, assigned by function (all mutants of one function are on the same side),
seeded, frozen in ``data/bugbench/manifest.json`` with the SHA-256 of ``mutants.jsonl``.
Re-building needs ``--force``. Do not use it (CLAUDE.md). Runs on eval need
``--i-have-approval-for-eval`` and are logged in ``data/bugbench/eval_runs.jsonl``.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time

from .datasets import log_eval_run
from .mutate import OPERATORS, Site, apply_site, mutation_sites
from .patch import _apply_file, parse, unified_diff
from .retrieval import frozen_sha256
from .sandbox import Sandbox, copy_selection
from .testrun import run_tests

BENCH_DIR = Path("data/bugbench")
GENERATOR_VERSION = "1"
DEFAULT_SEED = 20261108
DEFAULT_TARGETS = {"noema": 160, "more_itertools": 160}
MAX_MODULES = 3                   # a function mentioned by more test modules is too generic to target
MAX_SCOPE_SECONDS = 20.0          # test scopes slower than this on the base are skipped
MAX_PER_FUNCTION = 2


@dataclass(frozen=True)
class RepoSpec:
    name: str
    kind: str                     # git | vendored
    sources: tuple[str, ...]
    tests: str = "tests"
    path: str | None = None       # vendored: directory in this repository
    exclude: tuple[str, ...] = ()
    prefix: str = ""


REPOS = {
    "noema": RepoSpec("noema", "git", ("cpu_platform", "cpu_first"), prefix="nm",
                      exclude=("cpu_platform/sandbox.py", "cpu_platform/container.py", "cpu_platform/launch.py",
                               "cpu_platform/server.py", "cpu_platform/hardware.py", "cpu_platform/jobs.py")),
    "more_itertools": RepoSpec("more_itertools", "vendored", ("more_itertools",), prefix="mi",
                               path="data/bugbench/repos/more_itertools"),
}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# --------------------------------------------------------------------------- base repositories

def materialize(spec: RepoSpec, dest: Path, *, commit: str | None = None, root: Path = Path(".")) -> Path:
    """A pristine copy of the base repository in ``dest`` (which must not exist)."""
    dest.mkdir(parents=True)
    if spec.kind == "vendored":
        src = root / spec.path
        for rel in copy_selection(src):
            (dest / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src / rel, dest / rel)
        return dest
    if not commit or not re.fullmatch(r"[0-9a-f]{7,40}", commit):
        raise ValueError(f"{spec.name}: a commit id is required")
    done = subprocess.run(["git", "-c", "core.autocrlf=false", "-C", str(root), "archive", "--format=tar", commit],
                          capture_output=True, timeout=300)
    if done.returncode != 0:
        raise RuntimeError(f"git archive {commit} failed: {done.stderr.decode(errors='replace')[:300]} "
                           "(is the commit in this clone? run: git fetch origin main)")
    with tarfile.open(fileobj=io.BytesIO(done.stdout)) as tar:
        try:
            tar.extractall(dest, filter="data")
        except TypeError:                       # Python without extraction filters
            for member in tar.getmembers():
                if member.isfile() and not member.name.startswith(("/", "..")) and ".." not in member.name.split("/"):
                    tar.extract(member, dest)
    return dest


def source_files(spec: RepoSpec, base: Path) -> list[str]:
    out = []
    for top in spec.sources:
        for path in sorted((base / top).rglob("*.py")):
            rel = path.relative_to(base).as_posix()
            if rel not in spec.exclude and "/test" not in rel and not path.name.startswith("test_"):
                out.append(rel)
    return out


def test_modules(spec: RepoSpec, base: Path) -> dict[str, str]:
    """{module id usable as a test selector: source text}."""
    return {f"{spec.tests}.{p.stem}": p.read_text(encoding="utf-8", errors="replace")
            for p in sorted((base / spec.tests).glob("test*.py"))}


def scope_for(function: str, modules: dict[str, str]) -> list[str] | None:
    name = function.rsplit(".", 1)[-1]
    if name.startswith("__") or len(name) < 3:
        return None
    pattern = re.compile(rf"\b{re.escape(name)}\b")
    hits = [m for m, text in modules.items() if pattern.search(text)]
    return hits if 1 <= len(hits) <= MAX_MODULES else None


def _id(spec: RepoSpec, rel: str, site: Site) -> str:
    key = f"{spec.name}\0{rel}\0{site.start}\0{site.end}\0{site.replacement}"
    return f"{spec.prefix}-{hashlib.sha256(key.encode('utf-8')).hexdigest()[:10]}"


def candidate_stream(spec: RepoSpec, base: Path, seed: int):
    """(rel, function, site, scope) in a seeded order: functions shuffled, at most
    MAX_PER_FUNCTION sites per function, each time from the operator used least so far."""
    modules = test_modules(spec, base)
    groups = []
    for rel in source_files(spec, base):
        source = (base / rel).read_text(encoding="utf-8")
        by_function = defaultdict(list)
        for site in mutation_sites(source, rel):
            by_function[site.function].append(site)
        for function, sites in sorted(by_function.items()):
            scope = scope_for(function, modules)
            if scope:
                groups.append((rel, function, sites, scope))
    rng = random.Random(f"{seed}:{spec.name}")
    rng.shuffle(groups)
    used = Counter()
    for rel, function, sites, scope in groups:
        pool = list(sites)
        rng.shuffle(pool)
        taken = []
        for _ in range(min(MAX_PER_FUNCTION, len(pool))):
            pool.sort(key=lambda s: used[s.operator])           # stable: shuffled order breaks ties
            site = next((s for s in pool if s.operator not in {t.operator for t in taken}), pool[0])
            pool.remove(site)
            taken.append(site)
            used[site.operator] += 1
            yield rel, function, site, scope


# --------------------------------------------------------------------------- building

def build_repo(spec: RepoSpec, base: Path, target: int, seed: int, backend_factory, *, timeout: float = 60,
               log=print) -> tuple[list[dict], dict]:
    kept, stats = [], Counter()
    scope_ok: dict[tuple, bool] = {}
    box = Sandbox(base, backend_factory()).__enter__()
    try:
        for rel, function, site, scope in candidate_stream(spec, base, seed):
            if len(kept) >= target:
                break
            key = tuple(scope)
            if key not in scope_ok:
                started = time.perf_counter()
                run = run_tests(box, select=scope, timeout=timeout)
                scope_ok[key] = run.status == "passed" and time.perf_counter() - started <= MAX_SCOPE_SECONDS
                if not scope_ok[key]:
                    stats["scope_skipped"] += 1
                    log(f"  scope {scope} skipped on the base: {run.status}, {time.perf_counter() - started:.1f} s")
            if not scope_ok[key]:
                continue
            stats["tested"] += 1
            target_file = box.root / rel
            original = target_file.read_bytes()
            # Sites were computed on the text read with LF line endings; a CRLF checkout keeps
            # its endings in the mutated file.
            crlf = b"\r\n" in original
            source = original.decode("utf-8").replace("\r\n", "\n")
            mutated = apply_site(source, site)
            target_file.write_bytes((mutated.replace("\n", "\r\n") if crlf else mutated).encode("utf-8"))
            try:
                run = run_tests(box, select=scope, timeout=timeout)
            finally:
                target_file.write_bytes(original)
            changes = box.changes()
            if changes["modified"] or changes["deleted"]:
                stats["sandbox_rebuilt"] += 1           # the tests changed the copy: start a clean one
                box.__exit__(None, None, None)
                box = Sandbox(base, backend_factory()).__enter__()
            failing = sorted(t.id for t in run.tests if t.outcome in ("failed", "error"))
            if run.status != "failed" or not failing:
                stats["survived" if run.status == "passed" else f"discarded_{run.status}"] += 1
                continue
            stats["kept"] += 1
            kept.append({
                "id": _id(spec, rel, site), "repo": spec.name, "file": rel, "function": function,
                "operator": site.operator, "description": site.description, "line": site.line,
                "file_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
                "bug": unified_diff(rel, source, mutated), "reference_fix": unified_diff(rel, mutated, source),
                "test_scope": scope, "caught_by": failing[:20], "caught_count": len(failing),
            })
            if len(kept) % 20 == 0:
                log(f"  {spec.name}: {len(kept)} kept / {stats['tested']} tested")
    finally:
        box.__exit__(None, None, None)
    return kept, dict(stats)


def assign_splits(mutants: list[dict], seed: int) -> dict[str, list[str]]:
    """dev/eval by function, seeded, per repository: no function has mutants on both sides."""
    splits = {"dev": [], "eval": []}
    for repo in sorted({m["repo"] for m in mutants}):
        groups = sorted({(m["file"], m["function"]) for m in mutants if m["repo"] == repo})
        random.Random(f"{seed}:split:{repo}").shuffle(groups)
        side = {g: ("dev" if n % 2 == 0 else "eval") for n, g in enumerate(groups)}
        for m in mutants:
            if m["repo"] == repo:
                splits[side[(m["file"], m["function"])]].append(m["id"])
    return {k: sorted(v) for k, v in splits.items()}


def build(*, noema_commit: str, seed: int = DEFAULT_SEED, targets: dict | None = None, out_dir: Path = BENCH_DIR,
          backend_factory=None, force: bool = False, root: Path = Path("."), repos=None, log=print) -> dict:
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists() and not force:
        raise SystemExit(f"{manifest_path} exists: the benchmark is frozen. Rebuilding changes the eval split.")
    targets = targets or DEFAULT_TARGETS
    repos = repos or REPOS
    all_mutants, info = [], {}
    with tempfile.TemporaryDirectory(prefix="noema-bugbench-") as tmp:
        for name, target in targets.items():
            spec = repos[name]
            commit = noema_commit if spec.kind == "git" else None
            base = materialize(spec, Path(tmp) / name, commit=commit, root=root)
            log(f"{name}: building up to {target} mutants")
            started = time.perf_counter()
            kept, stats = build_repo(spec, base, target, seed, backend_factory, log=log)
            all_mutants += kept
            from .audit import tree_digest
            digest, files = tree_digest(base)
            info[name] = {"kind": spec.kind, "commit": commit, "path": spec.path, "tree_sha256": digest, "files": files,
                          "sources": list(spec.sources), "excluded": list(spec.exclude), "target": target,
                          "kept": len(kept), "stats": stats, "build_seconds": round(time.perf_counter() - started, 1),
                          "by_operator": dict(Counter(m["operator"] for m in kept))}
    all_mutants.sort(key=lambda m: m["id"])
    out_dir.mkdir(parents=True, exist_ok=True)
    mutants_path = out_dir / "mutants.jsonl"
    write_jsonl(mutants_path, all_mutants)
    splits = assign_splits(all_mutants, seed)
    backend = backend_factory().name
    manifest = {
        "schema": "noema-bugbench/1",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator": {"version": GENERATOR_VERSION, "seed": seed, "operators": list(OPERATORS),
                      "max_per_function": MAX_PER_FUNCTION, "max_modules": MAX_MODULES,
                      "max_scope_seconds": MAX_SCOPE_SECONDS, "build_backend": backend},
        "repos": info,
        "file": str(mutants_path).replace("\\", "/"),
        "sha256": frozen_sha256(mutants_path),
        "count": len(all_mutants),
        "split_seed": seed,
        "dev": splits["dev"],
        "eval": splits["eval"],
        "rule": "a mutant is kept only if the test modules that mention its function pass on the base and fail "
                "with the mutant; dev/eval are split by function",
        "use": {"dev": "tune prompts, context, budgets, search settings",
                "eval": "one run with settings frozen on dev; needs the user's approval (CLAUDE.md)"},
        "contamination_note": "more-itertools is public and likely in model training data; NOEMA is public on "
                              "GitHub. Mutants are new, but the surrounding code may be familiar.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


# --------------------------------------------------------------------------- loading and checking

def load(split: str | None = None, out_dir: Path = BENCH_DIR) -> tuple[dict, list[dict]]:
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    path = out_dir / "mutants.jsonl"
    if frozen_sha256(path) != manifest["sha256"]:
        raise SystemExit(f"{path} does not match its frozen manifest SHA-256; restore the original file")
    mutants = read_jsonl(path)
    if split:
        wanted = set(manifest[split])
        mutants = [m for m in mutants if m["id"] in wanted]
    return manifest, mutants


def spec_of(manifest: dict, name: str) -> RepoSpec:
    """The base repository as the frozen manifest describes it."""
    info = manifest["repos"][name]
    return RepoSpec(name, info["kind"], tuple(info["sources"]), path=info["path"], exclude=tuple(info["excluded"]))


def apply_bug(work: Path, mutant: dict) -> None:
    """Write the mutant into a pristine copy, checking the file is the one it was made from."""
    target = work / mutant["file"]
    text = target.read_bytes().decode("utf-8")
    if hashlib.sha256(text.replace("\r\n", "\n").encode("utf-8")).hexdigest() != mutant["file_sha256"]:
        raise RuntimeError(f"{mutant['id']}: {mutant['file']} differs from the file the mutant was made from")
    (fp,) = parse(mutant["bug"]).files
    new, _, error, notes = _apply_file(text, fp)
    if error or notes:
        raise RuntimeError(f"{mutant['id']}: the bug does not apply strictly: {error or notes}")
    target.write_bytes(new.encode("utf-8"))


def check(out_dir: Path = BENCH_DIR, root: Path = Path("."), *, run_tests_for: int = 0, backend_factory=None,
          seed: int = 0) -> list[str]:
    """Problems with the frozen benchmark ([] = fine): hashes, splits, every bug and reference
    fix applies; optionally re-runs the tests of a seeded sample of mutants."""
    manifest, mutants = load(None, out_dir)
    problems = []
    dev, ev = set(manifest["dev"]), set(manifest["eval"])
    if dev & ev:
        problems.append(f"dev and eval overlap: {sorted(dev & ev)[:5]}")
    if dev | ev != {m["id"] for m in mutants}:
        problems.append("splits do not cover exactly the mutants")
    side = {}
    for m in mutants:
        key = (m["repo"], m["file"], m["function"])
        s = "dev" if m["id"] in dev else "eval"
        if side.setdefault(key, s) != s:
            problems.append(f"function {key} has mutants in both splits")
    sample = set(random.Random(seed).sample([m["id"] for m in mutants], min(run_tests_for, len(mutants))))
    with tempfile.TemporaryDirectory(prefix="noema-bugbench-check-") as tmp:
        bases = {}
        for m in mutants:
            spec = spec_of(manifest, m["repo"])
            if m["repo"] not in bases:
                bases[m["repo"]] = materialize(spec, Path(tmp) / m["repo"], commit=manifest["repos"][m["repo"]]["commit"],
                                               root=root)
            base = bases[m["repo"]]
            original = (base / m["file"]).read_bytes()
            try:
                apply_bug(base, m)
                mutated = (base / m["file"]).read_bytes().decode("utf-8")
                (fp,) = parse(m["reference_fix"]).files
                restored, _, error, notes = _apply_file(mutated, fp)
                if error or notes or restored.replace("\r\n", "\n") != original.decode("utf-8").replace("\r\n", "\n"):
                    problems.append(f"{m['id']}: the reference fix does not restore the file")
                if m["id"] in sample:
                    with Sandbox(base, backend_factory()) as box:
                        run = run_tests(box, select=m["test_scope"], timeout=120)
                    failing = sorted(t.id for t in run.tests if t.outcome in ("failed", "error"))
                    if run.status != "failed" or failing[:20] != m["caught_by"]:
                        problems.append(f"{m['id']}: tests no longer catch it as recorded ({run.status}, {failing[:3]})")
            except RuntimeError as exc:
                problems.append(str(exc))
            finally:
                (base / m["file"]).write_bytes(original)
    return problems


# --------------------------------------------------------------------------- running the engine

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def run(split: str, *, model, backend_factory, candidates: int = 4, limit: int | None = None, out_path: Path,
        out_dir: Path = BENCH_DIR, root: Path = Path("."), allow_eval: bool = False, audit_dir: Path | None = None,
        max_output_tokens: int = 1024, ids: list[str] | None = None, log=print, search_factory=None) -> dict:
    from .contract import request_from_dict
    from .fix_search import FixSearch
    if split == "eval" and not allow_eval:
        raise SystemExit("the eval split is run once, with settings frozen on dev, and needs the user's approval: "
                         "pass --i-have-approval-for-eval")
    manifest, mutants = load(split, out_dir)
    if ids:
        mutants = [m for m in mutants if m["id"] in set(ids)]
    if limit:
        mutants = mutants[:limit]
    rows = []
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("", encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="noema-bugbench-run-") as tmp:
        bases = {}
        consecutive = 0
        for n, m in enumerate(mutants, 1):
            if m["repo"] not in bases:
                bases[m["repo"]] = materialize(spec_of(manifest, m["repo"]), Path(tmp) / "base" / m["repo"],
                                               commit=manifest["repos"][m["repo"]]["commit"], root=root)
            work = Path(tmp) / "work" / m["id"]
            shutil.copytree(bases[m["repo"]], work)
            apply_bug(work, m)
            mutated = (work / m["file"]).read_text(encoding="utf-8")
            request = request_from_dict({"question": f"Make the failing tests in {m['repo']} pass",
                                         "task": "repo_fix", "repo": {"path": str(work)}, "allow_execution": True,
                                         "client_request_id": m["id"],
                                         "budget": {"max_model_calls": candidates, "max_output_tokens": max_output_tokens}})
            search = (search_factory or (lambda: FixSearch(model, backend_factory, candidates=candidates)))()
            started = time.perf_counter()
            result = search.run(request, work, select=m["test_scope"], full_select=m["test_scope"],
                                index_db=Path(tmp) / "index" / f"{m['id']}.sqlite")
            exact = None
            if result.answer:
                parsed = parse(result.answer)
                if len(parsed.files) == 1 and parsed.files[0].path == m["file"]:
                    fixed, _, error, _ = _apply_file(mutated, parsed.files[0])
                    original = (bases[m["repo"]] / m["file"]).read_text(encoding="utf-8")
                    exact = error is None and fixed == original
                else:
                    exact = False
            caught = next((int(x.group(1)) for c in result.checked_claims
                           if (x := re.match(r"before any patch, (\d+) selected test", c.claim))), None)
            row = {"id": m["id"], "repo": m["repo"], "operator": m["operator"], "status": result.status,
                   "solved": result.status == "ok", "exact_restore": exact, "baseline_failing": caught,
                   "not_caught_here": caught == 0,
                   "candidates": [c.status for c in getattr(search, "candidates_tried", [])],
                   "model_calls": result.cost.model_calls, "prompt_tokens": result.cost.prompt_tokens,
                   "completion_tokens": result.cost.completion_tokens,
                   "model_seconds": result.cost.model_wall_seconds, "total_seconds": round(time.perf_counter() - started, 1),
                   "error": result.error.code if result.error else None,
                   "error_message": result.error.message[:500] if result.error else None, "answer": result.answer}
            rows.append(row)
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            if audit_dir:
                from .audit import write
                write(request, result, audit_dir)
            log(f"[{n}/{len(mutants)}] {m['id']} {m['operator']}: {row['status']}"
                + (" (exact restore)" if exact else "") + f", {row['model_calls']} call(s), {row['total_seconds']} s"
                + (f"\n    error {row['error']}: {row['error_message']}" if row["error"] else ""))
            shutil.rmtree(work, ignore_errors=True)
            model_error = result.status == "error" and result.error and result.error.code.startswith("local_model")
            consecutive = consecutive + 1 if model_error else 0
            # A timed-out request is one slow mutant (a long prompt on CPU): go on. A refused
            # connection, a model mismatch, or three model errors in a row mean the server is
            # not usable: stop.
            if model_error and ("timed out" not in result.error.message or consecutive >= 3):
                log("stopping: the model server is not usable (is it running and loaded? check "
                    "Invoke-WebRequest http://127.0.0.1:8080/health; a 503 'Loading model' means wait)")
                break
    summary = summarize(rows)
    summary.update(split=split, manifest_sha256=manifest["sha256"], candidates=candidates, results=str(out_path))
    if split == "eval":
        summary["eval_run_number"] = log_eval_run({"utc": datetime.now(timezone.utc).isoformat(), **summary}, out_dir)
    return summary


def excluded(row: dict) -> bool:
    """Not a task on this machine, so not in the rate: the mutant's tests did not fail here, or
    the baseline run could not be used (e.g. a mutant that exhausts memory is killed by the
    container's limit before any result is written). Both are reported by id."""
    return bool(row.get("not_caught_here")) or row.get("error") == "baseline_failed"


def summarize(rows: list[dict]) -> dict:
    def block(sel):
        # A mutant whose tests do not fail on this machine (for example a test skipped in the
        # container image) is not a task here: reported, and left out of the rate.
        sel = [r for r in sel if not excluded(r)]
        k, n = sum(r["solved"] for r in sel), len(sel)
        lo, hi = wilson(k, n)
        return {"solved": k, "attempted": n, "rate": round(k / n, 3) if n else None,
                "wilson95": [round(lo, 3), round(hi, 3)]}
    out = {"overall": block(rows), "by_repo": {}, "by_operator": {},
           "exact_restore": sum(1 for r in rows if r["exact_restore"]),
           "errors": sum(1 for r in rows if r["status"] == "error"),
           "not_caught_here": sorted(r["id"] for r in rows if r.get("not_caught_here")),
           "baseline_unusable": sorted(r["id"] for r in rows if r.get("error") == "baseline_failed"),
           "errors_counted": sum(1 for r in rows if r["status"] == "error" and not excluded(r)),
           "model_calls": sum(r["model_calls"] for r in rows),
           "model_seconds": round(sum(r["model_seconds"] for r in rows), 1),
           "total_seconds": round(sum(r["total_seconds"] for r in rows), 1)}
    for key, name in (("repo", "by_repo"), ("operator", "by_operator")):
        for value in sorted({r[key] for r in rows}):
            out[name][value] = block([r for r in rows if r[key] == value])
    return out


# --------------------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    from .sandbox import make_backend
    parser = argparse.ArgumentParser(description="NOEMA injected-bug benchmark (M4.1)")
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="generate, test and freeze the mutants (once)")
    b.add_argument("--noema-commit", required=True)
    b.add_argument("--seed", type=int, default=DEFAULT_SEED)
    b.add_argument("--target", action="append", default=[], metavar="REPO=N")
    b.add_argument("--force", action="store_true", help="rebuild a frozen benchmark (changes eval; do not)")
    c = sub.add_parser("check", help="verify the frozen benchmark")
    c.add_argument("--run-tests", type=int, default=0, metavar="N", help="also re-run the tests of N mutants")
    r = sub.add_parser("run", help="run fix search on mutants")
    r.add_argument("--split", choices=("dev", "eval"), required=True)
    r.add_argument("--limit", type=int)
    r.add_argument("--ids", nargs="*")
    r.add_argument("-n", "--candidates", type=int, default=4)
    r.add_argument("--model-url", default="http://127.0.0.1:8080")
    r.add_argument("--expect-model")
    r.add_argument("--max-output-tokens", type=int, default=1024)
    r.add_argument("--model-timeout", type=int, default=900, help="seconds per model request")
    r.add_argument("--out", default="bugbench_results.jsonl")
    r.add_argument("--audit", metavar="DIR")
    r.add_argument("--i-have-approval-for-eval", action="store_true", dest="allow_eval")
    sub.add_parser("status", help="counts per repository, operator and split")
    sm = sub.add_parser("summarize", help="recompute the summary from a results file (also a partial one)")
    sm.add_argument("results")
    for p in (b, c, r):
        p.add_argument("--backend", choices=("dev", "container"), default="container" if p is r else "dev")
        p.add_argument("--image")
        p.add_argument("--i-understand-dev-is-not-isolated", action="store_true", dest="ack")
    args = parser.parse_args(argv)
    if args.command == "summarize":
        rows = read_jsonl(Path(args.results))
        out = summarize(rows)
        out["rows"] = len(rows)
        print(json.dumps(out, indent=2))
        return 0
    if args.command == "status":
        manifest, mutants = load()
        side = {i: "dev" for i in manifest["dev"]} | {i: "eval" for i in manifest["eval"]}
        table = Counter((m["repo"], m["operator"], side[m["id"]]) for m in mutants)
        print(json.dumps({"count": manifest["count"], "sha256": manifest["sha256"],
                          "repos": {k: {"kept": v["kept"], "commit": v["commit"]} for k, v in manifest["repos"].items()},
                          "dev": len(manifest["dev"]), "eval": len(manifest["eval"]),
                          "table": {f"{r}/{o}/{s}": n for (r, o, s), n in sorted(table.items())}}, indent=2))
        return 0
    factory = lambda: make_backend(args.backend, args.ack, args.image)   # noqa: E731
    if args.command == "build":
        targets = dict(DEFAULT_TARGETS)
        for t in args.target:
            name, _, count = t.partition("=")
            targets[name] = int(count)
        manifest = build(noema_commit=args.noema_commit, seed=args.seed, targets=targets, backend_factory=factory,
                         force=args.force)
        print(json.dumps({k: manifest[k] for k in ("count", "sha256")} | {"repos": manifest["repos"]}, indent=2))
        return 0
    if args.command == "check":
        problems = check(run_tests_for=args.run_tests, backend_factory=factory)
        print(json.dumps({"ok": not problems, "problems": problems[:50]}, indent=2))
        return 0 if not problems else 1
    from .engine import LazyLocalModel
    expectation = None
    if args.expect_model:
        from .model_adapter import parse_expectation
        expectation = parse_expectation(args.expect_model)
    summary = run(args.split, model=LazyLocalModel(args.model_url, expectation=expectation, timeout=args.model_timeout),
                  backend_factory=factory,
                  candidates=args.candidates, limit=args.limit, ids=args.ids, out_path=Path(args.out),
                  allow_eval=args.allow_eval, audit_dir=Path(args.audit) if args.audit else None,
                  max_output_tokens=args.max_output_tokens)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
