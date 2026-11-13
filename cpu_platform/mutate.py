"""AST mutation operators for the injected-bug benchmark (roadmap M4.1). Standard library only.

    sites = mutation_sites(source, path)          # every candidate mutation in a file's functions
    mutated = apply_site(source, site)            # the source with one minimal edit

Each mutation is a small, local edit of the source text (the rest of the file is untouched, so
the diff is the bug and nothing else), made only inside function bodies, never in docstrings,
decorators, annotations or default values:

| Operator | What it does |
|---|---|
| ``flip_comparison`` | negates a comparison: ``<`` to ``>=``, ``==`` to ``!=``, ``in`` to ``not in``, ``is`` to ``is not`` (and back) |
| ``off_by_one`` | a boundary (``<`` to ``<=``, ``>`` to ``>=`` and back) or an integer used in a comparison, index, slice, ``range()`` or ``+``/``-`` changed by one |
| ``swapped_operands`` | swaps the operands of a non-commutative operator (``-``, ``/``, ``//``, ``%``, ``**``, ``<<``, ``>>``) |
| ``wrong_constant`` | ``True``/``False`` swapped, a float or an integer of magnitude >= 2 multiplied by 10, a short string literal that is compared, passed to a string method or returned changed |
| ``dropped_condition`` | one operand of ``and``/``or`` removed, or an ``if`` guard whose body only returns, raises, breaks or continues deleted |

Every mutated source must compile and parse to a different AST. Whether a mutant is a bug at
all is decided by the tests (``bugbench build`` keeps only mutants the tests catch).
"""
from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
import re

OPERATORS = ("flip_comparison", "off_by_one", "swapped_operands", "wrong_constant", "dropped_condition")
NEGATE = {"<": ">=", ">=": "<", ">": "<=", "<=": ">", "==": "!=", "!=": "==",
          "in": "not in", "not in": "in", "is": "is not", "is not": "is"}
BOUNDARY = {"<": "<=", "<=": "<", ">": ">=", ">=": ">"}
COMPARE_OP = re.compile(r"(?<![<>=!])(<=|>=|==|!=|<|>)(?![<>=])|\bnot\s+in\b|\bis\s+not\b|\bin\b|\bis\b")
NON_COMMUTATIVE = (ast.Sub, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.LShift, ast.RShift)
SIMPLE_STRING = re.compile(r"""^(['"])([^'"\\\n]{1,20})\1$""")
EARLY_EXIT = (ast.Return, ast.Raise, ast.Continue, ast.Break)


@dataclass(frozen=True)
class Site:
    operator: str
    function: str            # qualified name, e.g. "Class.method"
    line: int                # first line of the edited text (1-based)
    start: int               # character offsets into the source
    end: int
    original: str
    replacement: str
    description: str

    def to_dict(self) -> dict:
        return asdict(self)


class _Offsets:
    """Convert ast (line, UTF-8 byte column) positions to character offsets."""

    def __init__(self, source: str):
        self.lines = source.splitlines(keepends=True)
        self.starts = [0]
        for line in self.lines:
            self.starts.append(self.starts[-1] + len(line))

    def at(self, line: int, col: int) -> int:
        text = self.lines[line - 1]
        return self.starts[line - 1] + len(text.encode("utf-8")[:col].decode("utf-8", errors="replace"))

    def span(self, node) -> tuple[int, int]:
        return self.at(node.lineno, node.col_offset), self.at(node.end_lineno, node.end_col_offset)


def _body_nodes(fn) -> list:
    """Nodes inside a function's own body (not nested functions or classes, not docstrings)."""
    body = list(fn.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]
    out, stack = [], list(reversed(body))
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        out.append(node)
        stack.extend(reversed(list(ast.iter_child_nodes(node))))
    return out


def _functions(tree) -> list[tuple[str, ast.AST]]:
    out = []

    def visit(node, prefix):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = f"{prefix}{child.name}"
                out.append((name, child))
                visit(child, name + ".")
            elif isinstance(child, ast.ClassDef):
                visit(child, f"{prefix}{child.name}.")
            else:
                visit(child, prefix)
    visit(tree, "")
    return out


def _int_contexts(nodes) -> set[int]:
    """ids of integer constants used where off-by-one bugs live."""
    ids = set()
    for node in nodes:
        candidates = []
        if isinstance(node, ast.Compare):
            candidates = [node.left, *node.comparators]
        elif isinstance(node, ast.Subscript):
            candidates = [node.slice]
        elif isinstance(node, ast.Slice):
            candidates = [node.lower, node.upper, node.step]
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "range":
            candidates = list(node.args)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            candidates = [node.left, node.right]
        for c in candidates:
            if isinstance(c, ast.UnaryOp) and isinstance(c.op, ast.USub):
                c = c.operand
            if isinstance(c, ast.Constant) and type(c.value) is int:
                ids.add(id(c))
    return ids


STR_METHODS = {"startswith", "endswith", "split", "rsplit", "join", "strip", "rstrip", "lstrip", "replace",
               "partition", "rpartition", "find", "index", "count"}


def _str_contexts(nodes) -> set[int]:
    """ids of string constants whose value is likely behaviour, not a label or message:
    compared, passed to a string method, or returned."""
    ids = set()
    for node in nodes:
        if isinstance(node, ast.Compare):
            candidates = [node.left, *node.comparators]
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in STR_METHODS:
            candidates = [node.func.value, *node.args]
        elif isinstance(node, ast.Return):
            candidates = [node.value]
        else:
            continue
        for c in candidates:
            if isinstance(c, (ast.Tuple, ast.List, ast.Set)):
                candidates += list(c.elts)
            elif isinstance(c, ast.Constant) and type(c.value) is str:
                ids.add(id(c))
    return ids


def _in_fstrings(nodes) -> set[int]:
    return {id(v) for node in nodes if isinstance(node, ast.JoinedStr) for v in ast.walk(node) if v is not node}


def mutation_sites(source: str, path: str = "<file>") -> list[Site]:
    tree = ast.parse(source, path)
    offs = _Offsets(source)
    sites: list[Site] = []
    for qualname, fn in _functions(tree):
        nodes = _body_nodes(fn)
        int_ok, str_ok, skip = _int_contexts(nodes), _str_contexts(nodes), _in_fstrings(nodes)
        for node in nodes:
            if id(node) not in skip:
                sites += _sites_for(node, qualname, source, offs, int_ok, str_ok)
        sites += _guard_sites(fn, qualname, source, offs)
    valid, seen = [], set()
    for site in sites:
        key = (site.start, site.end, site.replacement)
        if key in seen:
            continue
        seen.add(key)
        if _changes_program(source, site, path):
            valid.append(site)
    return valid


def _sites_for(node, qualname, source, offs, int_ok, str_ok) -> list[Site]:
    out = []
    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        a = offs.at(node.left.end_lineno, node.left.end_col_offset)
        b = offs.at(node.comparators[0].lineno, node.comparators[0].col_offset)
        between = source[a:b]
        matches = list(COMPARE_OP.finditer(between))
        if len(matches) == 1:
            m = matches[0]
            op = re.sub(r"\s+", " ", m.group(0))
            start, end = a + m.start(), a + m.end()
            if op in NEGATE:
                out.append(Site("flip_comparison", qualname, node.lineno, start, end, m.group(0), NEGATE[op],
                                f"{op} -> {NEGATE[op]}"))
            if op in BOUNDARY:
                out.append(Site("off_by_one", qualname, node.lineno, start, end, m.group(0), BOUNDARY[op],
                                f"boundary {op} -> {BOUNDARY[op]}"))
    elif isinstance(node, ast.Constant):
        start, end = offs.span(node)
        text = source[start:end]
        value = node.value
        if type(value) is int and id(node) in int_ok and re.fullmatch(r"\d+", text):
            for delta in (1, -1):
                if value + delta >= 0:
                    out.append(Site("off_by_one", qualname, node.lineno, start, end, text, str(value + delta),
                                    f"{value} -> {value + delta}"))
        elif type(value) is bool:
            out.append(Site("wrong_constant", qualname, node.lineno, start, end, text, str(not value),
                            f"{value} -> {not value}"))
        elif type(value) is int and abs(value) >= 2 and re.fullmatch(r"\d+", text) and id(node) not in int_ok:
            out.append(Site("wrong_constant", qualname, node.lineno, start, end, text, str(value * 10),
                            f"{value} -> {value * 10}"))
        elif type(value) is float and re.fullmatch(r"\d+\.\d*|\d*\.\d+", text) and value != 0:
            new = repr(value * 10)
            out.append(Site("wrong_constant", qualname, node.lineno, start, end, text, new, f"{text} -> {new}"))
        elif type(value) is str and id(node) in str_ok and (m := SIMPLE_STRING.match(text)):
            q, body = m.group(1), m.group(2)
            new = f"{q}{body}_{q}"
            out.append(Site("wrong_constant", qualname, node.lineno, start, end, text, new, f"{text} -> {new}"))
    elif isinstance(node, ast.BinOp) and isinstance(node.op, NON_COMMUTATIVE):
        ls, le = offs.span(node.left)
        rs, re_ = offs.span(node.right)
        if source[ls:le].strip() != source[rs:re_].strip():
            replacement = source[rs:re_] + source[le:rs] + source[ls:le]
            out.append(Site("swapped_operands", qualname, node.lineno, ls, re_, source[ls:re_], replacement,
                            f"swap operands of {type(node.op).__name__}"))
    elif isinstance(node, ast.BoolOp) and len(node.values) >= 2:
        start, end = offs.span(node)
        word = " and " if isinstance(node.op, ast.And) else " or "
        parts = [source[slice(*offs.span(v))] for v in node.values]
        for i, dropped in enumerate(node.values):
            kept = parts[:i] + parts[i + 1:]
            replacement = kept[0] if len(kept) == 1 else word.join(kept)
            if len(kept) == 1 and isinstance(node.values[1 - i] if len(node.values) == 2 else None, ast.BoolOp):
                replacement = f"({replacement})"
            out.append(Site("dropped_condition", qualname, node.lineno, start, end, source[start:end], replacement,
                            f"drop operand {i + 1} of {word.strip()}"))
    return out


def _guard_sites(fn, qualname, source, offs) -> list[Site]:
    """``if cond: return/raise/...`` guards without else, in blocks with other statements.
    The guard must occupy whole lines, which are deleted."""
    out = []
    for block in _blocks(fn):
        if len(block) < 2:
            continue
        for stmt in block:
            if not (isinstance(stmt, ast.If) and not stmt.orelse and len(stmt.body) == 1
                    and isinstance(stmt.body[0], EARLY_EXIT)):
                continue
            line_start = offs.starts[stmt.lineno - 1]
            line_end = offs.starts[stmt.end_lineno]
            before = source[line_start:offs.at(stmt.lineno, stmt.col_offset)]
            after = source[offs.at(stmt.end_lineno, stmt.end_col_offset):line_end]
            if before.strip() or (after.strip() and not after.strip().startswith("#")):
                continue
            out.append(Site("dropped_condition", qualname, stmt.lineno, line_start, line_end,
                            source[line_start:line_end], "", f"delete guard `if {ast.unparse(stmt.test)[:60]}`"))
    return out


def _blocks(fn):
    stack = [fn]
    while stack:
        node = stack.pop()
        for name in ("body", "orelse", "finalbody"):
            block = getattr(node, name, None)
            if isinstance(block, list) and block and isinstance(block[0], ast.stmt):
                if node is not fn or name == "body":
                    yield block
                for stmt in block:
                    if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        stack.append(stmt)
        for handler in getattr(node, "handlers", []) or []:
            yield handler.body
            stack.extend(handler.body)


def apply_site(source: str, site: Site) -> str:
    if source[site.start:site.end] != site.original:
        raise ValueError("the source does not match the site (it changed since the sites were computed)")
    return source[:site.start] + site.replacement + source[site.end:]


def _changes_program(source: str, site: Site, path: str) -> bool:
    try:
        mutated = apply_site(source, site)
        compile(mutated, path, "exec", dont_inherit=True)
        return ast.dump(ast.parse(mutated)) != ast.dump(ast.parse(source))
    except (SyntaxError, ValueError):
        return False
