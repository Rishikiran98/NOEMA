"""Patch pipeline (roadmap M3.4): parse a model-proposed unified diff, validate it against a
policy, and apply it to a sandbox copy only. Standard library only.

    patch = parse(model_output)                 # also accepts ```diff fenced blocks
    check = validate(patch, box.root, PatchPolicy())
    if check.ok:
        applied = apply(box, patch)             # writes only inside the sandbox copy
        run = run_tests(box)                    # M3.3

    python -m cpu_platform.patch check FIX.diff --repo PATH
    python -m cpu_platform.patch try FIX.diff --repo PATH --backend container   # apply in a sandbox + run tests

Validation happens before any file is touched. Every problem is ``reject`` (the patch is not
applied) or ``flag`` (applied, but reported). What is checked:

* **form:** a parseable unified diff; renames, mode changes, symlinks and binary patches are
  rejected (never silently skipped); a file may appear once;
* **paths:** relative, no ``..``, no drive letters or UNC paths; never ``.git/``, CI configuration,
  likely-secret files, or files Python runs by itself at start-up (``sitecustomize.py``,
  ``usercustomize.py``, ``*.pth``), which could rig every test run;
* **tests:** test files, ``conftest.py`` and test configuration are protected. Changing them is
  rejected unless the policy allows it, and then still flagged (CLAUDE.md: the engine must
  restore passing tests without editing tests). Added code outside tests that reaches into the
  test machinery (``os._exit``, ``sys.exit`` monkeypatching, ``unittest``/``pytest`` patching) is flagged;
* **content:** added lines must be UTF-8 text without NUL bytes or invisible bidi/zero-width
  characters (a Trojan Source patch);
* **size:** files, changed lines and bytes per patch.

Applying requires every hunk's context to match the sandbox copy exactly (a unique line offset
is allowed, fuzz is not), computes every file before writing any, preserves line endings, and
then confirms with ``Sandbox.changes()`` that exactly the patched files changed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import fnmatch
import json
from pathlib import Path
import re
import sys

from .citation_check import safe_relative
from .injection import INVISIBLE
from .repo_index import secret_name_reason

HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
FENCE = re.compile(r"```(?:diff|patch|udiff)?[ \t]*\n(.*?)```", re.S)

TEST_PATTERNS = ("tests/*", "test/*", "*/tests/*", "*/test/*", "test_*.py", "*/test_*.py", "*_test.py",
                 "conftest.py", "*/conftest.py", "pytest.ini", "tox.ini", "noxfile.py")
TEST_CONFIG_MARKERS = {"pyproject.toml": "[tool.pytest", "setup.cfg": "[tool:pytest]"}
DENY_PATTERNS = (".git/*", ".github/*", ".gitlab-ci.yml", ".circleci/*", "Jenkinsfile", "azure-pipelines.yml",
                 ".noemaignore", "*sitecustomize.py", "*usercustomize.py", "*.pth")
TEST_MACHINERY = re.compile(r"\bos\._exit\b|\bsys\.exit\s*=|\bunittest\.(?:TestCase|TestResult|main)\s*[.=]"
                            r"|\bpytest\.\w+\s*=|\bbuiltins\.\w+\s*=|\bsys\.modules\[")


@dataclass
class Hunk:
    old_start: int
    old_len: int
    new_start: int
    new_len: int
    lines: list[str]               # each starts with ' ', '-' or '+'
    old_no_eol: bool = False       # "\ No newline at end of file" after an old line
    new_no_eol: bool = False


@dataclass
class FilePatch:
    path: str                      # normalized repo-relative path
    old_path: str | None           # raw header paths, for checks
    new_path: str | None
    new_file: bool = False
    deleted: bool = False
    hunks: list[Hunk] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)   # rename, mode change, binary, ...

    @property
    def added(self) -> int:
        return sum(1 for h in self.hunks for l in h.lines if l.startswith("+"))

    @property
    def removed(self) -> int:
        return sum(1 for h in self.hunks for l in h.lines if l.startswith("-"))


@dataclass
class Patch:
    files: list[FilePatch]
    text: str
    problems: list["Problem"] = field(default_factory=list)   # parse problems


@dataclass
class Problem:
    severity: str                  # reject | flag
    code: str
    path: str | None
    message: str


@dataclass
class PatchPolicy:
    allow: tuple[str, ...] = ()            # if set, every path must match one of these globs
    deny: tuple[str, ...] = ()             # extra globs to deny
    allow_test_changes: bool = False       # if True, test changes are flagged instead of rejected
    max_files: int = 10
    max_changed_lines: int = 400
    max_bytes: int = 200_000


@dataclass
class PatchCheck:
    ok: bool
    problems: list[Problem]
    files: list[dict]

    def to_dict(self) -> dict:
        return {"ok": self.ok, "problems": [asdict(p) for p in self.problems], "files": self.files}


@dataclass
class ApplyResult:
    applied: bool
    files: list[str]
    offsets: dict[str, list[int]]
    problems: list[Problem]
    changes: dict | None = None


# --------------------------------------------------------------------------- parsing

def extract_diff(text: str) -> str:
    """The diff in a model answer: fenced ```diff blocks (joined) if any, else the text from
    the first ``--- `` / ``diff --git`` line."""
    blocks = [b for b in FENCE.findall(text) if re.search(r"^(?:--- |diff --git |@@ )", b, re.M)]
    if blocks:
        return "\n".join(b.rstrip("\n") for b in blocks) + "\n"
    match = re.search(r"^(?:diff --git |--- )", text, re.M)
    return text[match.start():] if match else text


def _header_path(raw: str) -> str | None:
    raw = raw.split("\t", 1)[0].strip()
    if raw.startswith('"') and raw.endswith('"'):
        raw = raw[1:-1]
    if raw == "/dev/null":
        return None
    if raw.startswith(("a/", "b/")):
        raw = raw[2:]
    return raw


def parse(text: str) -> Patch:
    diff = extract_diff(text)
    lines = diff.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    files: list[FilePatch] = []
    problems: list[Problem] = []
    pending: list[str] = []          # git extended headers before ---/+++
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("diff --git "):
            pending = []
            i += 1
            while i < len(lines) and not lines[i].startswith(("--- ", "diff --git ")):
                pending.append(lines[i])
                i += 1
            for h in pending:
                if h.startswith(("rename from", "rename to", "copy from", "copy to")):
                    problems.append(Problem("reject", "rename", None, f"renames/copies are not supported: {h}"))
                elif h.startswith(("old mode", "new mode")) or (h.startswith(("new file mode", "deleted file mode"))
                                                                 and h.split()[-1] == "120000"):
                    problems.append(Problem("reject", "mode_change", None, f"mode changes/symlinks are not supported: {h}"))
                elif h.startswith("GIT binary patch") or h.startswith("Binary files"):
                    problems.append(Problem("reject", "binary", None, f"binary patches are not supported: {h}"))
            continue
        if line.startswith("Binary files ") or line.startswith("GIT binary patch"):
            problems.append(Problem("reject", "binary", None, f"binary patches are not supported: {line[:120]}"))
            i += 1
            continue
        if line.startswith("--- ") and i + 1 < len(lines) and lines[i + 1].startswith("+++ "):
            old, new = _header_path(line[4:]), _header_path(lines[i + 1][4:])
            path_raw = new if new is not None else old
            fp = FilePatch(path=path_raw or "", old_path=old, new_path=new,
                           new_file=old is None, deleted=new is None)
            if any(h.startswith("new file mode") for h in pending):
                fp.new_file = True
            if any(h.startswith("deleted file mode") for h in pending):
                fp.deleted = True
            if old is not None and new is not None and old != new:
                fp.unsupported.append(f"old and new paths differ ({old} -> {new}): renames are not supported")
            pending = []
            i += 2
            while i < len(lines) and lines[i].startswith("@@"):
                m = HUNK.match(lines[i])
                if not m:
                    problems.append(Problem("reject", "malformed", fp.path, f"bad hunk header: {lines[i][:80]}"))
                    break
                hunk = Hunk(int(m.group(1)), int(m.group(2) if m.group(2) is not None else 1),
                            int(m.group(3)), int(m.group(4) if m.group(4) is not None else 1), [])
                i += 1
                old_seen = new_seen = 0
                while i < len(lines) and (old_seen < hunk.old_len or new_seen < hunk.new_len
                                          or (i < len(lines) and lines[i].startswith("\\"))):
                    body = lines[i]
                    if body.startswith("\\"):
                        last = hunk.lines[-1][:1] if hunk.lines else " "
                        if last in (" ", "-"):
                            hunk.old_no_eol = True
                        if last in (" ", "+"):
                            hunk.new_no_eol = True
                        i += 1
                        continue
                    if body == "":
                        body = " "          # some tools drop the space on empty context lines
                    tag = body[0]
                    if tag not in " -+":
                        break
                    hunk.lines.append(body)
                    old_seen += tag in " -"
                    new_seen += tag in " +"
                    i += 1
                if (old_seen, new_seen) != (hunk.old_len, hunk.new_len):
                    problems.append(Problem("reject", "malformed", fp.path,
                                            f"hunk counts do not match its header (-{hunk.old_len} +{hunk.new_len}, "
                                            f"found -{old_seen} +{new_seen})"))
                fp.hunks.append(hunk)
            files.append(fp)
            continue
        i += 1
    if not files and not problems:
        problems.append(Problem("reject", "empty", None, "no unified diff found"))
    return Patch(files, diff, problems)


# --------------------------------------------------------------------------- validation

def _matches(path: str, patterns) -> bool:
    return any(fnmatch.fnmatch(path, p) for p in patterns)


def is_test_path(path: str, root: Path | None = None) -> bool:
    if _matches(path, TEST_PATTERNS):
        return True
    marker = TEST_CONFIG_MARKERS.get(path)
    if marker and root is not None and (root / path).is_file():
        try:
            return marker in (root / path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return False
    return False


def validate(patch: Patch, root: str | Path, policy: PatchPolicy = PatchPolicy()) -> PatchCheck:
    root = Path(root)
    problems = list(patch.problems)
    raw_bytes = patch.text.encode("utf-8", errors="surrogateescape")
    if b"\x00" in raw_bytes:
        problems.append(Problem("reject", "binary", None, "the diff contains NUL bytes"))
    if len(raw_bytes) > policy.max_bytes:
        problems.append(Problem("reject", "too_large", None, f"diff is {len(raw_bytes)} bytes (limit {policy.max_bytes})"))
    if len(patch.files) > policy.max_files:
        problems.append(Problem("reject", "too_many_files", None,
                                f"{len(patch.files)} files (limit {policy.max_files})"))
    changed = sum(f.added + f.removed for f in patch.files)
    if changed > policy.max_changed_lines:
        problems.append(Problem("reject", "too_large", None, f"{changed} changed lines (limit {policy.max_changed_lines})"))
    seen: set[str] = set()
    summaries = []
    for fp in patch.files:
        raw_paths = [p for p in (fp.old_path, fp.new_path) if p is not None]
        safe = [safe_relative(p) for p in raw_paths]
        if not raw_paths or any(s is None for s in safe) or any(p != s for p, s in zip(raw_paths, safe)):
            problems.append(Problem("reject", "path", fp.path or None,
                                    "path is absolute, escapes the repository or is not normalized"))
            continue
        path = safe[0]
        fp.path = path
        for reason in fp.unsupported:
            problems.append(Problem("reject", "rename", path, reason))
        if path in seen:
            problems.append(Problem("reject", "duplicate", path, "the file appears more than once in the patch"))
        seen.add(path)
        if _matches(path, DENY_PATTERNS) or _matches(path, policy.deny):
            problems.append(Problem("reject", "denied_path", path, "this path may not be changed by a patch"))
        if secret_name_reason(path.rsplit("/", 1)[-1]):
            problems.append(Problem("reject", "secret_path", path, "likely-secret files may not be changed"))
        if policy.allow and not _matches(path, policy.allow):
            problems.append(Problem("reject", "not_allowed", path, "outside the allowed paths"))
        if is_test_path(path, root):
            severity = "flag" if policy.allow_test_changes else "reject"
            problems.append(Problem(severity, "touches_tests", path,
                                    "changes a test file or test configuration (the engine must make tests pass "
                                    "without editing them)"))
        target = root / path
        if target.is_symlink():
            problems.append(Problem("reject", "symlink", path, "the target is a symlink"))
        if fp.new_file and target.exists():
            problems.append(Problem("reject", "exists", path, "patch creates a file that already exists"))
        if not fp.new_file and not target.is_file():
            problems.append(Problem("reject", "missing", path, "patch changes a file that does not exist"))
        test_file = is_test_path(path, root)
        for h in fp.hunks:
            for line in h.lines:
                if not line.startswith("+"):
                    continue
                added = line[1:]
                if INVISIBLE.search(added):
                    problems.append(Problem("reject", "invisible_characters", path,
                                            "an added line contains bidi/zero-width characters (Trojan Source)"))
                    break
                if "�" in added:
                    problems.append(Problem("reject", "binary", path, "an added line is not valid UTF-8 text"))
                    break
                if not test_file and TEST_MACHINERY.search(added):
                    problems.append(Problem("flag", "test_machinery", path,
                                            f"added code reaches into the test machinery: {added.strip()[:100]}"))
        summaries.append({"path": path, "added": fp.added, "removed": fp.removed,
                          "new_file": fp.new_file, "deleted": fp.deleted})
    # One finding per (code, path) is enough.
    unique, keys = [], set()
    for p in problems:
        if (p.severity, p.code, p.path) not in keys:
            keys.add((p.severity, p.code, p.path))
            unique.append(p)
    return PatchCheck(not any(p.severity == "reject" for p in unique), unique, summaries)


# --------------------------------------------------------------------------- applying

def _split(text: str) -> tuple[list[str], str, bool]:
    newline = "\r\n" if "\r\n" in text else "\n"
    ends_with_newline = text.endswith(("\n", "\r\n"))
    body = text[:-len(newline)] if ends_with_newline and text.endswith(newline) else text
    return (body.split(newline) if text else []), newline, ends_with_newline


def _apply_file(old_text: str, fp: FilePatch) -> tuple[str | None, list[int], str | None]:
    """(new text, offsets, error)."""
    lines, newline, eol = _split(old_text)
    out: list[str] = []
    cursor = 0
    offsets = []
    final_eol = eol
    for h in fp.hunks:
        old_block = [l[1:] for l in h.lines if l[0] in " -"]
        new_block = [l[1:] for l in h.lines if l[0] in " +"]
        want = max(h.old_start - 1, 0) if h.old_len else h.old_start
        candidates = [want] if lines[want:want + len(old_block)] == old_block else []
        if not candidates:
            candidates = [s for s in range(cursor, len(lines) - len(old_block) + 1)
                          if lines[s:s + len(old_block)] == old_block]
            if len(candidates) != 1:
                reason = "context not found" if not candidates else "context matches more than one place"
                return None, offsets, f"hunk @@ -{h.old_start},{h.old_len} does not apply: {reason}"
        start = candidates[0]
        if start < cursor:
            return None, offsets, f"hunk @@ -{h.old_start},{h.old_len} overlaps the previous hunk"
        offsets.append(start - want)
        out += lines[cursor:start] + new_block
        cursor = start + len(old_block)
        if cursor == len(lines):
            # This hunk reaches the end of the file: the new side's "\ No newline at end of
            # file" marker alone decides how the file ends (new files included).
            final_eol = not h.new_no_eol
    out += lines[cursor:]
    if fp.deleted:
        return "", offsets, None if not out else "deletion patch does not remove the whole file"
    text = newline.join(out)
    if out and final_eol:
        text += newline
    if fp.new_file and not out:
        text = ""
    return text, offsets, None


def apply(box, patch: Patch, policy: PatchPolicy = PatchPolicy()) -> ApplyResult:
    """Validate, then apply to the sandbox copy (``box.root``) atomically. Never touches the
    original repository: it only accepts a Sandbox."""
    check = validate(patch, box.root, policy)
    if not check.ok:
        return ApplyResult(False, [], {}, check.problems)
    root = box.root.resolve()
    planned: dict[str, str | None] = {}
    offsets: dict[str, list[int]] = {}
    for fp in patch.files:
        target = (box.root / fp.path).resolve()
        if not target.is_relative_to(root):
            return ApplyResult(False, [], {}, check.problems + [Problem("reject", "path", fp.path, "escapes the copy")])
        old = "" if fp.new_file else target.read_bytes().decode("utf-8", errors="strict")
        new, offs, error = _apply_file(old, fp)
        if error:
            return ApplyResult(False, [], {}, check.problems + [Problem("reject", "does_not_apply", fp.path, error)])
        planned[fp.path] = None if fp.deleted else new
        offsets[fp.path] = offs
    for path, content in planned.items():     # nothing is written until every file applied
        target = box.root / path
        if content is None:
            target.unlink()
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content.encode("utf-8"))
    changes = box.changes()
    touched = set(changes["created"]) | set(changes["modified"]) | set(changes["deleted"])
    problems = list(check.problems)
    unexpected = touched - set(planned)
    if unexpected:
        problems.append(Problem("flag", "other_changes", None,
                                f"the sandbox copy also differs in files the patch did not name: {sorted(unexpected)[:10]}"))
    return ApplyResult(True, sorted(planned), offsets, problems, changes)


# --------------------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check or try a unified diff against a repository")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "try"):
        p = sub.add_parser(name)
        p.add_argument("patch", help="diff file, or a model answer containing ```diff blocks")
        p.add_argument("--repo", required=True)
        p.add_argument("--allow-test-changes", action="store_true")
        if name == "try":
            p.add_argument("--backend", choices=("dev", "container"), default="dev")
            p.add_argument("--image")
            p.add_argument("--i-understand-dev-is-not-isolated", action="store_true", dest="ack")
            p.add_argument("--select", nargs="*", default=[])
    args = parser.parse_args(argv)
    patch = parse(Path(args.patch).read_text(encoding="utf-8", errors="replace"))
    policy = PatchPolicy(allow_test_changes=args.allow_test_changes)
    if args.command == "check":
        result = validate(patch, Path(args.repo), policy)
        print(json.dumps(result.to_dict(), indent=2))
        return 0 if result.ok else 1
    from .sandbox import Sandbox, SandboxError, make_backend
    from .testrun import run_tests
    try:
        with Sandbox(args.repo, make_backend(args.backend, args.ack, args.image)) as box:
            applied = apply(box, patch, policy)
            out = {"applied": applied.applied, "files": applied.files, "offsets": applied.offsets,
                   "problems": [asdict(p) for p in applied.problems]}
            if applied.applied:
                run = run_tests(box, select=args.select)
                out["tests"] = {"status": run.status, "counts": run.counts, "problems": run.problems,
                                "failing": [t.id for t in run.tests if t.outcome in ("failed", "error")],
                                "backend": run.record.backend, "security_boundary": run.record.security_boundary,
                                "trust_note": run.trust_note}
    except SandboxError as exc:
        parser.error(str(exc))
    print(json.dumps(out, indent=2))
    return 0 if out["applied"] and out.get("tests", {}).get("status") == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
