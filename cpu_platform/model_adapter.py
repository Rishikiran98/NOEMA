"""llama.cpp model adapter: check that the served model is the one expected (roadmap M1.4).

v0.10 served whatever model ``/v1/models`` listed first, with no check of identity, context
size or thinking support (audit G5). This module reads what the server reports and compares
it with an operator's ``ModelExpectation`` before any request is served:

* identity: a preset name, or name fragments, must occur in the model id or file name;
* context: the server's ``n_ctx`` must be at least ``min_ctx``; unknown counts as a failure;
* file hash (optional): SHA256 of the GGUF the server actually loaded;
* thinking: what the model supports, recorded for per-request checks in ``LocalLlama``.

Failing closed: if something the expectation asks about cannot be read, that is a problem,
not a pass. Standard library only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
from urllib.request import Request

from cpu_first.core import (LocalLlama, ModelMismatch, ModelRejectedRequest, model_http_json,
                            model_server_headers, thinking_support)
from .hardware import MODEL_PRESETS


@dataclass(frozen=True)
class ModelExpectation:
    """What the operator expects to be loaded. Every field is optional."""
    name_fragments: tuple[str, ...] = ()   # all must occur in the model id or file name
    min_ctx: int | None = None
    sha256: str | None = None
    thinking: bool | None = None           # declared support; checked against the name
    label: str = "custom"

    def __post_init__(self):
        if self.min_ctx is not None and (type(self.min_ctx) is not int or self.min_ctx < 1):
            raise ValueError("min_ctx must be a positive integer")
        if self.sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ValueError("sha256 must be 64 lowercase hex characters")
        if any(not f.strip() for f in self.name_fragments):
            raise ValueError("name fragments must be non-empty")


@dataclass(frozen=True)
class ModelIdentity:
    """What the server reports. None = the server did not say."""
    id: str
    model_path: str | None
    n_ctx: int | None
    build_info: str | None
    model_count: int
    thinking_supported: bool | None
    sha256: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _normalize(text: str) -> str:
    return re.sub(r"[-_.\s]+", "", text.lower())


def expectation_for_preset(name: str) -> ModelExpectation:
    """Expect a launcher preset: its repository name and quantization, its preferred
    context (if any) and its declared thinking support."""
    preset = MODEL_PRESETS.get(name)
    if preset is None:
        raise ValueError(f"unknown preset {name!r}; choose from {', '.join(MODEL_PRESETS)}")
    repo, _, quant = preset.hf.partition(":")
    stem = repo.rsplit("/", 1)[-1]
    stem = re.sub(r"-GGUF$", "", stem, flags=re.I)
    fragments = (stem, quant) if quant else (stem,)
    return ModelExpectation(name_fragments=fragments, min_ctx=preset.ctx_size,
                            thinking=preset.thinking, label=name)


def parse_expectation(value: str, *, min_ctx: int | None = None,
                      sha256: str | None = None) -> ModelExpectation:
    """``--expect-model`` value: a preset name, or comma-separated name fragments."""
    if value in MODEL_PRESETS:
        base = expectation_for_preset(value)
        return ModelExpectation(base.name_fragments, min_ctx if min_ctx is not None else base.min_ctx,
                                sha256, base.thinking, base.label)
    fragments = tuple(f.strip() for f in value.split(",") if f.strip())
    if not fragments:
        raise ValueError("--expect-model needs a preset name or name fragments")
    return ModelExpectation(fragments, min_ctx, sha256, None, value)


def probe(base_url: str, *, timeout: float = 10, hash_file: bool = False) -> ModelIdentity:
    """Read /v1/models and /props from a loopback llama.cpp server. Transport failures
    raise ModelTransportError (retryable); a malformed answer raises ModelProtocolError."""
    client = LocalLlama(base_url, timeout=int(max(1, timeout)))
    try:
        props = model_http_json(Request(client.base_url + "/props", headers=model_server_headers()), timeout)
    except ModelRejectedRequest as exc:
        if exc.status != 404:  # 401 etc. are real errors; only "no such endpoint" is tolerated
            raise
        props = {}             # very old builds: context and path stay unknown (fail closed later)
    settings = props.get("default_generation_settings")
    n_ctx = settings.get("n_ctx") if isinstance(settings, dict) else None
    if n_ctx is None:
        n_ctx = props.get("n_ctx")
    path = props.get("model_path")
    digest = None
    if hash_file and isinstance(path, str) and Path(path).is_file():
        from .provenance import file_sha256
        digest = file_sha256(Path(path))
    return ModelIdentity(
        id=client.model,
        model_path=path if isinstance(path, str) else None,
        n_ctx=n_ctx if type(n_ctx) is int else None,
        build_info=props.get("build_info") if isinstance(props.get("build_info"), str) else None,
        model_count=client.model_count,
        thinking_supported=client.thinking_supported,
        sha256=digest,
    )


def problems(identity: ModelIdentity, expect: ModelExpectation) -> list[str]:
    """Every way the served model differs from the expectation (empty = matches)."""
    out = []
    if identity.model_count > 1:
        out.append(f"server lists {identity.model_count} models; NOEMA expects exactly one "
                   "(a router or proxy may answer with a different model)")
    names = [identity.id] + ([Path(identity.model_path).name] if identity.model_path else [])
    for fragment in expect.name_fragments:
        if not any(_normalize(fragment) in _normalize(n) for n in names):
            out.append(f"model {identity.id!r} does not match expected {expect.label!r} "
                       f"(missing {fragment!r})")
    if expect.min_ctx is not None:
        if identity.n_ctx is None:
            out.append(f"context size unknown (server /props did not report n_ctx); "
                       f"expected at least {expect.min_ctx}")
        elif identity.n_ctx < expect.min_ctx:
            out.append(f"server context {identity.n_ctx} is below the expected {expect.min_ctx}; "
                       f"relaunch with --ctx-size {expect.min_ctx}")
    if expect.sha256 is not None:
        if identity.sha256 is None:
            out.append("model file SHA256 could not be computed (model_path not reported or not "
                       "readable from this process)")
        elif identity.sha256 != expect.sha256:
            out.append(f"model file SHA256 {identity.sha256[:12]}... differs from the expected "
                       f"{expect.sha256[:12]}...")
    if expect.thinking is not None and identity.thinking_supported is not None \
            and expect.thinking != identity.thinking_supported:
        out.append(f"expected thinking support {expect.thinking}, but {identity.id!r} "
                   f"{'supports' if identity.thinking_supported else 'does not support'} it")
    return out


def verify(base_url: str, expect: ModelExpectation, *, timeout: float = 10) -> ModelIdentity:
    """Probe and compare; raise ModelMismatch listing every problem."""
    identity = probe(base_url, timeout=timeout, hash_file=expect.sha256 is not None)
    found = problems(identity, expect)
    if found:
        raise ModelMismatch(found)
    return identity
