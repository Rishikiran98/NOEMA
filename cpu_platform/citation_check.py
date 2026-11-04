"""Citation verifier (roadmap M2.4): checks the citations in a model answer against the
repository's working tree, deterministically, and says exactly what each check shows.

For every citation ``path:START-END`` (or ``[E<n>:...]``, mapped by ``repo_qa``):

1. **path**: relative, no ``..``, resolves inside the repository, not through a symlink.
   Anything else is refuted as an invalid path and never touched.
2. **file**: only files the index accepted are read. A cited file that the index skipped
   (secret, binary, too large) or ignores is **not opened**: reading it to check a quote
   would let an answer probe its contents. Such citations are labelled, not dropped.
3. **lines**: ``1 <= START <= END <= number of lines``.
4. **quotes**: code quoted in the same sentence as the citation, as inline backticks or a
   fenced block, must occur in the cited lines. Inline quotes must match exactly; fenced
   blocks may differ only in common indentation and trailing spaces.
5. **relevance**: whether the cited code supports the statement is **NOT CHECKED**, ever.
   A citation that exists and quotes correctly is reported as "exists, relevance NOT CHECKED".

Results become ``CheckedClaim`` entries (checkers ``citation_check.file``,
``citation_check.lines``, ``citation_check.quote``), each about one specific fact.

    python -m cpu_platform.citation_check verify RESULT.json --repo PATH   # re-check a saved result
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import re
import stat
import sys
import textwrap

from .contract import CheckedClaim, EngineResult, Evidence, NotRun, UnverifiedClaim, result_from_dict, validate_result

INLINE_CODE = re.compile(r"(?<!`)`([^`\n]+)`(?!`)")
FENCE = re.compile(r"```[^\n]*\n(.*?)```", re.S)
MIN_QUOTE_CHARS = 2
SCOPE_FILE = "the cited file exists in the repository's working tree and was indexed; says nothing about its relevance"
SCOPE_LINES = "the cited line range lies within the file; says nothing about what those lines do"
SCOPE_QUOTE = ("the quoted code occurs in the cited lines (inline: exact; fenced: ignoring common indentation "
               "and trailing spaces); says nothing about whether it supports the statement")


@dataclass
class QuoteCheck:
    quote: str
    kind: str          # inline | fenced
    found: bool


@dataclass
class CitationCheck:
    path: str
    start_line: int
    end_line: int
    status: str        # verified | refuted | not_checked
    reasons: list[str] = field(default_factory=list)
    file_ok: bool | None = None
    lines_ok: bool | None = None
    quotes: list[QuoteCheck] = field(default_factory=list)
    span_sha256: str | None = None
    unknown_excerpt: bool = False
    file_lines: int | None = None
    file_sha256: str | None = None

    @property
    def label(self) -> str:
        return f"{self.path}:{self.start_line}-{self.end_line}"


# --------------------------------------------------------------------------- quotes

def _normalize_block(text: str) -> list[str]:
    lines = [line.rstrip() for line in textwrap.dedent(text).splitlines()]
    while lines and not lines[0]:
        lines.pop(0)
    while lines and not lines[-1]:
        lines.pop()
    return lines


def quote_found(kind: str, quote: str, span_lines: list[str]) -> bool:
    if kind == "inline":
        return quote.strip() in "\n".join(span_lines)
    want = _normalize_block(quote)
    if not want:
        return True
    # Slide a window over the cited lines; each window is dedented on its own, so a quote
    # copied from inside an indented block still matches.
    for i in range(len(span_lines) - len(want) + 1):
        if _normalize_block("\n".join(span_lines[i:i + len(want)])) == want:
            return True
    return False


# --------------------------------------------------------------------------- files

def safe_relative(path: str) -> str | None:
    """A normalized repo-relative posix path, or None if it is absolute or escapes."""
    if not path or "\x00" in path:
        return None
    if path.startswith(("/", "\\")) or PureWindowsPath(path).drive or re.match(r"^[A-Za-z]:", path):
        return None
    parts = [p for p in path.replace("\\", "/").split("/") if p not in ("", ".")]
    if not parts or any(p == ".." for p in parts):
        return None
    return "/".join(parts)


class CitationChecker:
    """Checks citations against one repository. ``file_status`` maps indexed paths to
    (status, skip_reason, sha256), as recorded by the index."""

    def __init__(self, root: str | Path, file_status: dict[str, tuple]):
        self.root = Path(root).resolve()
        self.file_status = file_status
        self._lines: dict[str, list[str] | None] = {}

    def _read(self, rel: str) -> list[str] | None:
        if rel not in self._lines:
            try:
                data = (self.root / rel).read_bytes()
                self._lines[rel] = data.decode("utf-8-sig").splitlines()
            except (OSError, UnicodeDecodeError):
                self._lines[rel] = None
        return self._lines[rel]

    def _through_symlink(self, rel: str) -> bool:
        current = self.root
        for part in rel.split("/"):
            current = current / part
            try:
                if stat.S_ISLNK(os.lstat(current).st_mode):
                    return True
            except OSError:
                return False
        return False

    def check(self, path: str, start: int, end: int, quotes: list[tuple[str, str]] = (),
              excerpt_sha256: str | None = None) -> CitationCheck:
        rel = safe_relative(path)
        if rel is None:
            return CitationCheck(path, start, end, "refuted", ["invalid path: absolute or outside the repository"],
                                 file_ok=False)
        result = CitationCheck(rel, start, end, "verified")
        target = self.root / rel
        try:
            inside = target.resolve().is_relative_to(self.root)
        except (OSError, ValueError):
            inside = False
        if not inside or self._through_symlink(rel):
            return replace(result, status="refuted", file_ok=False,
                           reasons=["invalid path: resolves outside the repository or through a symlink"])
        status = self.file_status.get(rel)
        if status is None:
            if os.path.lexists(target):
                return replace(result, status="not_checked", reasons=[
                    "file exists but is not in the index (ignored or never indexed); not read, so lines "
                    "and quotes NOT CHECKED"])
            return replace(result, status="refuted", file_ok=False, reasons=["file does not exist"])
        if status[0] != "indexed":
            return replace(result, status="not_checked", reasons=[
                f"file was excluded from the index ({status[1]}); deliberately not read, so lines and "
                "quotes NOT CHECKED"])
        lines = self._read(rel)
        if lines is None:
            return replace(result, status="refuted", file_ok=False,
                           reasons=["file no longer exists or is unreadable in the working tree"])
        result.file_ok = True
        result.file_lines = len(lines)
        result.file_sha256 = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
        if not (1 <= start <= end <= len(lines)):
            result.lines_ok = False
            result.status = "refuted"
            result.reasons.append(f"line range {start}-{end} is outside the file (1-{len(lines)})")
            return result
        result.lines_ok = True
        span = lines[start - 1:end]
        result.span_sha256 = hashlib.sha256("\n".join(span).encode("utf-8")).hexdigest()
        for kind, quote in quotes:
            if len(quote.strip()) < MIN_QUOTE_CHARS:
                continue
            result.quotes.append(QuoteCheck(quote, kind, quote_found(kind, quote, span)))
        if any(not q.found for q in result.quotes):
            result.status = "refuted"
            bad = [q.quote for q in result.quotes if not q.found]
            result.reasons.append(f"{len(bad)} quoted snippet(s) not found in the cited lines")
        result.reasons.append("relevance NOT CHECKED")
        return result


# --------------------------------------------------------------------------- result integration

def _quote_spans(answer: str) -> list[tuple[int, int, str, str]]:
    """(start, end, kind, quote) for fenced blocks and inline code outside fences."""
    spans = [(m.start(), m.end(), "fenced", m.group(1)) for m in FENCE.finditer(answer)]
    for m in INLINE_CODE.finditer(answer):
        if not any(a <= m.start() < b for a, b, _, _ in spans):
            spans.append((m.start(), m.end(), "inline", m.group(1)))
    return sorted(spans)


def _sentence_bounds(answer: str, code: list[tuple[int, int, str, str]]) -> list[int]:
    """Positions where a sentence ends, ignoring '.' inside code."""
    cuts = [m.start() for m in re.finditer(r"(?<=[.!?])\s+|\n\s*\n", answer)
            if not any(a <= m.start() < b for a, b, _, _ in code)]
    return cuts + [len(answer)]


def assign_quotes(answer: str, citations) -> dict[int, list[tuple[str, str]]]:
    """Map citation index -> quotes it vouches for. Within a sentence, a quote belongs to
    the next citation after it; quotes after the sentence's last citation belong to that
    one. Quotes in sentences without a citation belong to no citation (not checked)."""
    code = _quote_spans(answer)
    bounds = _sentence_bounds(answer, code)

    def sentence_of(pos: int) -> int:
        return next(i for i, b in enumerate(bounds) if pos <= b)

    occurrences = []   # (position, citation index)
    for n, cite in enumerate(citations):
        start = 0
        while True:
            pos = answer.find(cite.text, start)
            if pos < 0:
                break
            if not any(a <= pos < b for a, b, _, _ in code):
                occurrences.append((pos, n))
            start = pos + 1
    occurrences.sort()
    assigned: dict[int, list[tuple[str, str]]] = {n: [] for n in range(len(citations))}
    for a, _, kind, quote in code:
        sentence = sentence_of(a)
        same = [(pos, n) for pos, n in occurrences if sentence_of(pos) == sentence]
        if not same:
            continue
        after = [n for pos, n in same if pos > a]
        owner = after[0] if after else same[-1][1]
        if (kind, quote) not in assigned[owner]:
            assigned[owner].append((kind, quote))
    return assigned


def check_answer(answer: str, citations, checker: CitationChecker) -> list[CitationCheck]:
    """Check each parsed ``repo_qa.Citation`` with the quotes that precede it in its clause."""
    quotes = assign_quotes(answer, citations)
    out = []
    for n, cite in enumerate(citations):
        if cite.unknown_excerpt:     # [E9]: an excerpt id that was never shown
            out.append(CitationCheck(cite.path, 0, 0, "refuted", ["names an excerpt id that was not shown to the model"],
                                     file_ok=False, unknown_excerpt=True))
            continue
        out.append(checker.check(cite.path, cite.start_line, cite.end_line, quotes[n]))
    return out


def apply_to_result(result: EngineResult, checks: list[CitationCheck]) -> EngineResult:
    """Add checked claims (and current-source evidence) for each citation check, and
    relabel each citation's unverified claim with what was and was not established."""
    evidence = list(result.evidence)
    checked = list(result.checked_claims)
    known = {e.id for e in evidence} | {x.id for x in result.executions}
    by_label: dict[str, CitationCheck] = {}
    for n, c in enumerate(checks, 1):
        label = f"[{c.path}]" if c.unknown_excerpt else c.label
        by_label[label] = c
        if c.status == "not_checked":
            continue
        if c.file_ok:
            ev_id = f"C{n}"
            while ev_id in known:
                ev_id += "x"
            known.add(ev_id)
            if c.lines_ok:
                evidence.append(Evidence(id=ev_id, kind="file_span", source=c.path, start_line=c.start_line,
                                         end_line=c.end_line, sha256=c.span_sha256, retrieved_by="citation_check"))
            elif c.file_lines:   # the whole file, which shows the cited range does not exist
                evidence.append(Evidence(id=ev_id, kind="file_span", source=c.path, start_line=1,
                                         end_line=c.file_lines, sha256=c.file_sha256, retrieved_by="citation_check"))
            else:                # empty file: nothing to span
                evidence.append(Evidence(id=ev_id, kind="input", source=c.path, retrieved_by="citation_check"))
            basis = [ev_id]
        else:
            basis = ["answer"] if "answer" in known else [next(iter(known))]
        checked.append(CheckedClaim(f"cited file {c.path} exists", "verified" if c.file_ok else "refuted",
                                    "citation_check.file", SCOPE_FILE, basis,
                                    [] if c.file_ok else c.reasons))
        if c.file_ok:
            checked.append(CheckedClaim(f"lines {c.start_line}-{c.end_line} exist in {c.path}",
                                        "verified" if c.lines_ok else "refuted", "citation_check.lines",
                                        SCOPE_LINES, basis, [] if c.lines_ok else c.reasons[:1]))
        for q in c.quotes:
            checked.append(CheckedClaim(f"`{q.quote[:300]}` appears in {c.label}", "verified" if q.found else "refuted",
                                        "citation_check.quote", SCOPE_QUOTE, basis,
                                        [f"{q.kind} quote"]))
    unverified = []
    for claim in result.unverified_claims:
        if claim.claim.startswith("citation "):
            check = by_label.get(claim.claim[len("citation "):])
            if check is not None:
                summary = _summary(check)
                claim = UnverifiedClaim(claim.claim, f"{summary}; {claim.reason}"[:2000], claim.evidence_ids)
        unverified.append(claim)
    not_run = [n for n in result.not_run if n.item != "citation verification"]
    if checks and not any(n.item == "citation relevance" for n in not_run):
        not_run.append(NotRun("citation relevance", "whether a cited span supports the statement is NOT CHECKED"))
    return validate_result(replace(result, evidence=evidence, checked_claims=checked,
                                   unverified_claims=unverified, not_run=not_run))


def _summary(c: CitationCheck) -> str:
    if c.status == "not_checked":
        return "citation check: " + c.reasons[0]
    if not c.file_ok:
        return "citation check: REFUTED (" + "; ".join(c.reasons) + ")"
    if not c.lines_ok:
        return "citation check: file exists, line range REFUTED (" + c.reasons[0] + ")"
    quotes = (f", {sum(q.found for q in c.quotes)}/{len(c.quotes)} quotes match" if c.quotes else ", no quotes")
    head = "citation check: exists" + quotes
    if any(not q.found for q in c.quotes):
        head += " (quote REFUTED)"
    return head + ", relevance NOT CHECKED"


def checker_for(retriever) -> CitationChecker:
    return CitationChecker(retriever.root, retriever.file_status)


def main(argv: list[str] | None = None) -> int:
    from .repo_index import RepoIndex
    from .repo_qa import parse_citations
    from .retrieval import Hit
    parser = argparse.ArgumentParser(description="Re-check the citations of a saved repo_ask EngineResult")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("verify")
    p.add_argument("result", help="EngineResult JSON file (from repo_qa --format json)")
    p.add_argument("--repo", required=True)
    p.add_argument("--db")
    args = parser.parse_args(argv)
    data = json.loads(Path(args.result).read_text(encoding="utf-8"))
    original = result_from_dict(data, strict=False)
    if not original.answer:
        parser.error("the result has no answer to check")
    shown = {e.id: Hit(e.id, e.source, e.start_line, e.end_line, "window", None, 0.0, [])
             for e in original.evidence if e.kind == "file_span" and e.id.startswith("E")}
    with RepoIndex(args.repo, args.db) as index:
        index.update()
        status = {f["path"]: (f["status"], f["skip_reason"], f["sha256"]) for f in index.files()}
    checks = check_answer(original.answer, parse_citations(original.answer, shown), CitationChecker(args.repo, status))
    out = [{"citation": c.label, "status": c.status, "reasons": c.reasons,
            "quotes": [{"quote": q.quote, "kind": q.kind, "found": q.found} for q in c.quotes]} for c in checks]
    print(json.dumps(out, indent=2))
    return 0 if all(c.status != "refuted" for c in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
