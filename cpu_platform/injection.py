"""Prompt-injection hardening for repository text (roadmap M2.5). Standard library only.

Repository content is untrusted: a comment, docstring or README can address the model
("ignore previous instructions ..."). This module makes two kinds of guarantee:

* **Structural, enforced here:** text from the repository is passed to the model only as
  delimited data, and cannot forge structure. ``neutralize`` defuses chat-template control
  tokens (``<|im_start|>``, ``[INST]``, ``<<SYS>>``, ``<s>``) that a server could turn into
  real role markers, ``<think>`` tags that would change how the answer is parsed, and
  invisible bidirectional/zero-width characters ("Trojan Source"), each replaced by a
  visible marker. Excerpt delimiters are escaped in ``repo_qa``.
* **Detection, reported not enforced:** ``instruction_like`` flags text that addresses the
  model. Flagged excerpts are kept (dropping them would let an attacker hide code by
  adding a magic comment) and reported in the result. ``echoes`` flags an answer that
  repeats flagged text.

Whether a model *obeys* injected text is behaviour, not structure: it is measured with the
canary harness (``python -m cpu_platform.injection_eval``), never assumed.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

# Chat-template and role markers used by common local models (ChatML/Qwen, Llama 2/3,
# Mistral, Gemma, Phi, DeepSeek) plus generic role headers.
CONTROL_TOKENS = re.compile(
    r"<\|[A-Za-z0-9_\-]{1,40}\|>"                  # <|im_start|> <|eot_id|> <|endoftext|> <|system|>
    r"|<｜[^｜<>]{1,40}｜>"                          # DeepSeek fullwidth bars
    r"|\[/?INST\]|<<\s*/?\s*SYS\s*>>"              # Llama 2 / Mistral
    r"|</?s>"                                       # BOS/EOS written as text
    r"|<(?:start|end)_of_turn>"                     # Gemma
    r"|</?think>",                                  # reasoning tags parsed by the client
    re.I)
# Bidirectional overrides/isolates and zero-width characters (CVE-2021-42574 "Trojan Source").
INVISIBLE = re.compile("[‪-‮⁦-⁩​-‏⁠﻿]")

INSTRUCTION_PATTERNS = (
    ("ignore_instructions", re.compile(
        r"\b(?:ignore|disregard|forget|override|bypass)\b[^.\n]{0,40}\b(?:previous|prior|above|earlier|all|any|system|your)\b"
        r"[^.\n]{0,25}\b(?:instruction|prompt|rule|direction|guideline)s?\b", re.I)),
    ("role_reassignment", re.compile(
        r"\b(?:you are now|from now on,? you|pretend to be|you must now|new instructions?)\b"
        r"|\bact as (?:if you|an? (?:AI|assistant|different|new|unrestricted|helpful))\b", re.I)),
    ("address_to_model", re.compile(
        r"\b(?:AI|assistant|language model|LLM|chatbot|model)\s*[:,]\s*(?:please\s+)?(?:ignore|reply|respond|answer|say|print|output|tell)\b"
        r"|\b(?:note|message|instruction)s?\s+(?:to|for)\s+(?:the\s+)?(?:AI|assistant|LLM|language model|model)\b", re.I)),
    ("output_directive", re.compile(
        r"\b(?:reply|respond|say|print|output)\s+(?:only\s+)?(?:with|exactly)\b[^.\n]{0,60}"
        r"|\banswer\s+(?:only|exactly)\s+with\b[^.\n]{0,60}"
        r"|\b(?:say|print|output|include|mention|add|write)\b[^.\n]{0,60}\bin (?:your|the) (?:answer|reply|response|output)\b"
        r"|\b(?:output|print|say|repeat) (?:it|this|that)\b(?: first| now| verbatim)?", re.I)),
    ("exfiltration", re.compile(
        r"\b(?:reveal|leak|disclose|exfiltrate)\b[^.\n]{0,30}\b(?:system prompt|secret|password|api key|token|credential)s?\b"
        r"|\b(?:print|output|show|send|tell me)\s+(?:me\s+)?(?:the|your|any|all)\s+(?:\w+\s+){0,2}(?:system prompt|secrets?|passwords?|api keys?|(?:access|auth|bearer|api) tokens?|credentials?)\b", re.I)),
    ("prompt_reference", re.compile(r"\bsystem prompt\b|\bprevious instructions\b", re.I)),
)


@dataclass(frozen=True)
class Finding:
    label: str
    line: int          # 1-based within the text examined
    snippet: str       # the matched text, truncated


def neutralize(text: str) -> tuple[str, list[str]]:
    """Return (text with control tokens and invisible characters made inert and visible,
    list of what was changed). Content stays readable: ``<|im_start|>`` becomes
    ``⟦|im_start|⟧`` and U+202E becomes ``⟦U+202E⟧``, which no chat template treats as markup."""
    changes: list[str] = []

    def token(match: re.Match) -> str:
        changes.append(f"control token {match.group(0)!r}")
        inner = match.group(0)
        return "⟦" + inner.replace("<", "").replace(">", "").replace("[", "").replace("]", "") + "⟧"

    def invisible(match: re.Match) -> str:
        changes.append(f"invisible character U+{ord(match.group(0)):04X}")
        return f"⟦U+{ord(match.group(0)):04X}⟧"

    text = CONTROL_TOKENS.sub(token, text)
    text = INVISIBLE.sub(invisible, text)
    return text, changes


def instruction_like(text: str) -> list[Finding]:
    """Lines that look like they address the model. A heuristic: it flags, it never blocks."""
    out = []
    for n, line in enumerate(text.splitlines(), 1):
        for label, pattern in INSTRUCTION_PATTERNS:
            match = pattern.search(line)
            if match:
                out.append(Finding(label, n, match.group(0)[:120]))
                break
    return out


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


def echoes(answer: str, flagged_lines: list[str], min_words: int = 5) -> list[str]:
    """Flagged lines whose wording (any run of ``min_words`` consecutive words) reappears
    in the answer: a sign the answer was steered by, or is quoting, injected text."""
    answer_words = _words(answer)
    grams = {tuple(answer_words[i:i + min_words]) for i in range(len(answer_words) - min_words + 1)}
    hits = []
    for line in flagged_lines:
        words = _words(line)
        if any(tuple(words[i:i + min_words]) in grams for i in range(len(words) - min_words + 1)):
            hits.append(line.strip()[:200])
    return hits
