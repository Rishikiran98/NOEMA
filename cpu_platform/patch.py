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

Models write imperfect diffs, so reading and placing them is lenient in a few bounded ways,
each reported as a ``flag`` (never silent): hunk header counts that disagree with the body are
recounted from the body (like ``git apply --recount``); a header without line numbers is placed
by its context alone; and a hunk whose context does not match exactly may be placed with up to
FUZZ unchanged context lines ignored at each end, then with whitespace differences inside lines
ignored (indentation still exact). A lenient placement must match exactly one place, removed
lines are never ignored, and unchanged lines are always taken from the file, not the model.
Applying computes every file before writing any, preserves line endings, and then confirms with
``Sandbox.changes()`` that exactly the patched files changed.
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
BARE_HUNK = re.compile(r"^@@(?: [^@\d-][^@]*| \.\.\.)? ?@@")   # "@@ @@", "@@ ... @@" (no numbers)
FUZZ = 2                           # context lines that may be trimmed from each end of a hunk
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
    located: bool = True           # False: the header had no line numbers


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
    notes: list[Problem] = []        # flags: the diff was read leniently (see patch.md)
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
                if m:
                    hunk = Hunk(int(m.group(1)), int(m.group(2) if m.group(2) is not None else 1),
                                int(m.group(3)), int(m.group(4) if m.group(4) is not None else 1), [])
                elif BARE_HUNK.match(lines[i]):
                    # "@@ @@" or "@@ ... @@": no line numbers. The hunk is placed by its context
                    # alone (it must match exactly one place), and that is flagged.
                    hunk = Hunk(0, -1, 0, -1, [], located=False)
                    notes.append(Problem("flag", "no_line_numbers", fp.path,
                                         "hunk header has no line numbers; placed by its context alone"))
                else:
                    problems.append(Problem("reject", "malformed", fp.path, f"bad hunk header: {lines[i][:80]}"))
                    break
                i += 1
                # The body runs to the next hunk/file header or the first line that is not a
                # hunk line. Header counts are checked against it, not used to cut it short: a
                # header that undercounts must not silently drop changed lines.
                while i < len(lines) and not lines[i].startswith(("@@", "diff --git ")) \
                        and not (lines[i].startswith("--- ") and i + 1 < len(lines) and lines[i + 1].startswith("+++ ")):
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
                    if body[0] not in " -+":
                        break
                    hunk.lines.append(body)
                    i += 1
                # Blank lines between the diff and the prose after it read as empty context lines;
                # drop trailing ones the header does not count (all of them when it has no counts).
                while hunk.lines and hunk.lines[-1] == " " and (
                        hunk.old_len < 0 or sum(1 for l in hunk.lines if l[0] in " -") > hunk.old_len):
                    hunk.lines.pop()
                old_seen = sum(1 for l in hunk.lines if l[0] in " -")
                new_seen = sum(1 for l in hunk.lines if l[0] in " +")
                if not any(l[0] in "-+" for l in hunk.lines):
                    problems.append(Problem("reject", "malformed", fp.path, "hunk has no changed lines"))
                elif hunk.old_len >= 0 and (old_seen, new_seen) != (hunk.old_len, hunk.new_len):
                    # Models often miscount. Like `git apply --recount`, trust the body, and say so.
                    notes.append(Problem("flag", "recounted", fp.path,
                                         f"hunk header said -{hunk.old_len} +{hunk.new_len}, body has "
                                         f"-{old_seen} +{new_seen}; the body was used"))
                hunk.old_len, hunk.new_len = old_seen, new_seen
                fp.hunks.append(hunk)
            files.append(fp)
            continue
        i += 1
    if not files and not problems:
        problems.append(Problem("reject", "empty", None, "no unified diff found"))
    return Patch(files, diff, problems + notes)


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


def _loose(line: str) -> str:
    """Whitespace-insensitive form of a line: leading indentation kept exactly (it is syntax in
    Python), other runs of whitespace collapsed, trailing whitespace dropped."""
    body = line.lstrip(" \t")
    return line[:len(line) - len(body)] + " ".join(body.split())


def _attempts(h: Hunk):
    """(hunk lines, trimmed context lines, loose) to try, strictest first. Only unchanged
    context lines are ever trimmed (at most FUZZ from each end); changed lines never are."""
    lead = next((k for k, l in enumerate(h.lines) if l[0] != " "), len(h.lines))
    trail = next((k for k, l in enumerate(reversed(h.lines)) if l[0] != " "), len(h.lines))
    trims = sorted({(a, b) for a in range(min(lead, FUZZ) + 1) for b in range(min(trail, FUZZ) + 1)},
                   key=lambda t: (t[0] + t[1], t))
    for loose in (False, True):
        for a, b in trims:
            yield h.lines[a:len(h.lines) - b], a, a + b, loose


def _apply_file(old_text: str, fp: FilePatch) -> tuple[str | None, list[int], str | None, list[str]]:
    """(new text, offsets, error, notes). Each hunk is placed exactly if possible; otherwise
    with up to FUZZ context lines trimmed from each end, then whitespace-insensitively, and only
    where it then matches exactly one place. Every lenient placement is reported in notes."""
    lines, newline, eol = _split(old_text)
    loose_lines = [_loose(l) for l in lines]
    out: list[str] = []
    cursor = 0
    offsets = []
    notes: list[str] = []
    final_eol = eol
    for h in fp.hunks:
        placed = None
        for body, lead, trimmed, loose in _attempts(h):
            old_block = [l[1:] for l in body if l[0] in " -"]
            if trimmed and not old_block:
                continue
            haystack, needle = (loose_lines, [_loose(l) for l in old_block]) if loose else (lines, old_block)
            lead_old = sum(1 for l in h.lines[:lead] if l[0] in " -")
            want = (max(h.old_start - 1, 0) if h.old_len else h.old_start) + lead_old
            # A strict hunk may sit where its header says even if its context recurs elsewhere;
            # a lenient one must match exactly one place in the rest of the file.
            if h.located and not (trimmed or loose) and haystack[want:want + len(needle)] == needle \
                    and want >= cursor:
                candidates = [want]
            elif not needle:
                candidates = []
            else:
                candidates = [s for s in range(cursor, len(lines) - len(needle) + 1)
                              if haystack[s:s + len(needle)] == needle]
            if len(candidates) == 1:
                placed = (body, candidates[0], want, trimmed, loose, len(old_block))
                break
            if len(candidates) > 1 and not trimmed and not loose:
                return None, offsets, f"hunk @@ -{h.old_start},{h.old_len} does not apply: context matches more than one place", notes
        if placed is None:
            return None, offsets, f"hunk @@ -{h.old_start},{h.old_len} does not apply: context not found", notes
        body, start, want, trimmed, loose, old_count = placed
        if start < cursor:
            return None, offsets, f"hunk @@ -{h.old_start},{h.old_len} overlaps the previous hunk", notes
        if trimmed or loose:
            notes.append(f"hunk @@ -{h.old_start},{h.old_len} placed leniently at line {start + 1}"
                         + (f": {trimmed} context line(s) ignored" if trimmed else "")
                         + ("; whitespace differences ignored" if loose else ""))
        offsets.append(start - want if h.located else 0)
        out += lines[cursor:start]
        k = start
        for l in body:                  # unchanged lines come from the file, not from the model
            if l[0] == " ":
                out.append(lines[k])
                k += 1
            elif l[0] == "-":
                k += 1
            else:
                out.append(l[1:])
        cursor = start + old_count
        if cursor == len(lines):
            # This hunk reaches the end of the file: the new side's "\ No newline at end of
            # file" marker alone decides how the file ends (new files included).
            final_eol = not h.new_no_eol
    out += lines[cursor:]
    if fp.deleted:
        return "", offsets, (None if not out else "deletion patch does not remove the whole file"), notes
    text = newline.join(out)
    if out and final_eol:
        text += newline
    if fp.new_file and not out:
        text = ""
    return text, offsets, None, notes


def apply(box, patch: Patch, policy: PatchPolicy = PatchPolicy()) -> ApplyResult:
    """Validate, then apply to the sandbox copy (``box.root``) atomically. Never touches the
    original repository: it only accepts a Sandbox."""
    check = validate(patch, box.root, policy)
    if not check.ok:
        return ApplyResult(False, [], {}, check.problems)
    root = box.root.resolve()
    planned: dict[str, str | None] = {}
    offsets: dict[str, list[int]] = {}
    fuzz: list[Problem] = []
    for fp in patch.files:
        target = (box.root / fp.path).resolve()
        if not target.is_relative_to(root):
            return ApplyResult(False, [], {}, check.problems + [Problem("reject", "path", fp.path, "escapes the copy")])
        old = "" if fp.new_file else target.read_bytes().decode("utf-8", errors="strict")
        new, offs, error, notes = _apply_file(old, fp)
        if error:
            return ApplyResult(False, [], {}, check.problems + [Problem("reject", "does_not_apply", fp.path, error)])
        planned[fp.path] = None if fp.deleted else new
        offsets[fp.path] = offs
        fuzz += [Problem("flag", "fuzzy_context", fp.path, n) for n in notes]
    for path, content in planned.items():     # nothing is written until every file applied
        target = box.root / path
        if content is None:
            target.unlink()
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content.encode("utf-8"))
    changes = box.changes()
    touched = set(changes["created"]) | set(changes["modified"]) | set(changes["deleted"])
    problems = list(check.problems) + fuzz
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
