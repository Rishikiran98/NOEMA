"""gitignore-compatible path matching, standard library only (roadmap M2.1).

Implements the rules in gitignore(5) that matter for indexing a working tree:

* blank lines and ``#`` comments are ignored; ``\\#`` and ``\\!`` escape; trailing
  unescaped spaces are stripped;
* ``!pattern`` re-includes a path excluded by an earlier pattern (but a file inside an
  excluded directory stays excluded, because the walker never enters that directory);
* a trailing ``/`` matches directories only;
* a pattern with a ``/`` at the start or in the middle is anchored to the directory of
  its ``.gitignore``; otherwise it matches a name at any depth below it;
* ``*`` and ``?`` do not match ``/``; ``[...]`` is a character class (``[!...]`` negated);
  leading ``**/``, trailing ``/**`` and inner ``/**/`` span directories; any other
  ``**`` behaves like ``*``;
* the last matching pattern wins, and patterns from deeper ``.gitignore`` files are
  consulted after (so they override) those from parent directories.

Tests cross-check this module against ``git check-ignore`` when git is installed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class Rule:
    base: str            # directory the rule is relative to, posix, "" = repo root
    pattern: str         # original text, for explanations
    regex: re.Pattern
    negate: bool
    dir_only: bool
    source: str          # file the rule came from (or "<default>")

    def matches(self, rel: str, is_dir: bool) -> bool:
        if self.dir_only and not is_dir:
            return False
        if self.base:
            if not rel.startswith(self.base + "/"):
                return False
            rel = rel[len(self.base) + 1:]
        return self.regex.fullmatch(rel) is not None


def _translate(pattern: str) -> str:
    """Translate one gitignore glob (no leading '!' or '/', no trailing '/') to a regex."""
    out, i, n = [], 0, len(pattern)
    while i < n:
        if pattern.startswith("**/", i) and (i == 0 or pattern[i - 1] == "/"):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("/**", i) and i + 3 == n:
            out.append("/.*")
            i += 3
        elif pattern[i] == "*":
            while i < n and pattern[i] == "*":   # '**' not at a boundary acts like '*'
                i += 1
            out.append("[^/]*")
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        elif pattern[i] == "[":
            j = i + 1
            if j < n and pattern[j] in "!^":
                j += 1
            if j < n and pattern[j] == "]":
                j += 1
            while j < n and pattern[j] != "]":
                j += 1
            if j >= n:                          # unterminated: a literal '['
                out.append(re.escape("["))
                i += 1
                continue
            body = pattern[i + 1:j]
            negated = body[:1] in ("!", "^")
            if negated:
                body = body[1:]
            body = body.replace("\\", "\\\\")
            out.append("[" + ("^/" if negated else "") + body + "]")
            i = j + 1
        elif pattern[i] == "\\" and i + 1 < n:
            out.append(re.escape(pattern[i + 1]))
            i += 2
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return "".join(out)


def parse_line(line: str, base: str = "", source: str = "<default>") -> Rule | None:
    """One gitignore line -> Rule, or None for blanks and comments."""
    line = line.rstrip("\n").rstrip("\r")
    # Strip trailing spaces unless escaped with a backslash.
    while line.endswith(" ") and not line.endswith("\\ "):
        line = line[:-1]
    if line.endswith("\\ "):
        line = line[:-2] + " "
    if not line or line.startswith("#"):
        return None
    negate = False
    if line.startswith("!"):
        negate, line = True, line[1:]
    elif line.startswith("\\!") or line.startswith("\\#"):
        line = line[1:]
    dir_only = line.endswith("/") and not line.endswith("\\/")
    if dir_only:
        line = line.rstrip("/")
    if not line:
        return None
    anchored = "/" in line
    if line.startswith("/"):
        line = line[1:]
    body = _translate(line)
    regex = re.compile(body if anchored else "(?:.*/)?" + body, re.S)
    return Rule(base, ("!" if negate else "") + line + ("/" if dir_only else ""), regex, negate, dir_only, source)


def parse_lines(lines, base: str = "", source: str = "<default>") -> list[Rule]:
    return [r for r in (parse_line(l, base, source) for l in lines) if r is not None]


class IgnoreMatcher:
    """Accumulates rules while a walker descends; ``ignored`` applies the last match."""

    def __init__(self, rules: list[Rule] | None = None):
        self.rules: list[Rule] = list(rules or [])

    def add_file(self, path: Path, base: str, *, source: str | None = None) -> int:
        """Add rules from an ignore file; unreadable or non-UTF-8 files add nothing."""
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return 0
        rules = parse_lines(text.splitlines(), base, source or str(path))
        self.rules.extend(rules)
        return len(rules)

    def explain(self, rel: str, is_dir: bool) -> Rule | None:
        """The rule that decides ``rel`` (None = no rule matched, i.e. not ignored)."""
        decision = None
        for rule in self.rules:
            if rule.matches(rel, is_dir):
                decision = rule
        return decision

    def ignored(self, rel: str, is_dir: bool) -> bool:
        rule = self.explain(rel, is_dir)
        return rule is not None and not rule.negate

    def child(self) -> "IgnoreMatcher":
        return IgnoreMatcher(self.rules)
