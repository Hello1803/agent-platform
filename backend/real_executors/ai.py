"""
Real executors for the AI-category node types (classify, extract_information,
summarize, make_decision, ai_agent, ask_ai), replacing the stubs in
backend/compiler/executors.py with actual LLM calls.

The interesting problem here isn't calling an API -- it's that a spec
author (a human via the canvas, or the NL compiler) defines what a
`classify` node's labels are, or what fields an `extract_information` node
should pull out, at SPEC-AUTHORING time, not at code-writing time. So the
structured-output schema these executors ask for has to be built
DYNAMICALLY from `node.config`, per node, per call -- see
`_build_extract_schema` and `_build_classify_schema` below.

Every function here returns a NodeExecutor (the same `(node, spec,
variables) -> dict` signature the stubs use), built against an
`AIProvider` (backend/ai_providers/base.py) rather than a concrete SDK
client -- this module has no idea whether it's talking to Anthropic,
OpenAI, or anything else, and doesn't need to. See
backend/ai_providers/registry.py for how the concrete provider actually
gets chosen and its credential resolved, both server-side.
"""

from __future__ import annotations

import json
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, create_model

from backend.ai_providers.base import AIProvider
from backend.ai_providers.registry import build_provider
from backend.compiler.executors import NodeExecutor
from backend.models.agent_spec import AgentNode, AgentSpecification, NodeType

def _default_max_tokens() -> int:
    import os
    try:
        return max(256, int(os.environ.get("AI_MAX_TOKENS", "4096")))
    except ValueError:
        return 4096


DEFAULT_MAX_TOKENS = _default_max_tokens()  # was 1024 — long answers were truncated


_FIELD_TYPE_MAP: dict[str, type] = {"string": str, "number": float, "boolean": bool}


def _variables_context(variables: dict[str, Any], *, max_chars: int = 12000, max_rows: int = 25) -> str:
    """Compact workflow state for the model prompt.

    Read File can put hundreds of rows in `rows`; Code may add a filtered
    `table_preview`. Dumping the full bag blows local LLM context windows.
    Prefer small previews and hard-cap list length / total size.
    """
    prefer_keys = (
        "rag_context",
        "rag_chunks",
        "rag_hit_count",
        "table_preview",
        "filtered_rows",
        "filtered_students",
        "row_count",
        "table_columns",
        "table_row_count",
        "table_summary",
        "current_item",
        "ai_response",
        "draft_email",
        "summary",
        "email",
        "emails",
        "api_response",
        "api_status",
        "code_ok",
        "file_path",
        "to",
        "recipient_email",
    )
    skip_keys = {"file_contents", "file_bytes_base64", "rows"}  # rows often huge; use table_preview

    slim: dict[str, Any] = {}
    # Prefer curated keys first
    for key in prefer_keys:
        if key in variables and variables[key] is not None:
            slim[key] = variables[key]
    # Then other small scalars / short values
    for key, value in variables.items():
        if key in slim or key in skip_keys:
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            if isinstance(value, str) and len(value) > 2000:
                slim[key] = value[:2000] + "…(truncated)"
            else:
                slim[key] = value
        elif isinstance(value, list) and len(value) <= 5:
            slim[key] = value
        elif isinstance(value, dict) and len(json.dumps(value, default=str)) < 1500:
            slim[key] = value

    def _trim(obj: Any) -> Any:
        if isinstance(obj, list):
            trimmed = obj[:max_rows]
            trimmed = [_trim(x) for x in trimmed]
            if len(obj) > max_rows:
                return trimmed + [f"…({len(obj) - max_rows} more rows omitted)"]
            return trimmed
        if isinstance(obj, dict):
            return {k: _trim(v) for k, v in obj.items()}
        if isinstance(obj, str) and len(obj) > 4000:
            return obj[:4000] + "…(truncated)"
        return obj

    slim = _trim(slim)
    # If table_preview missing but rows exist, take a sample only
    if "table_preview" not in slim and isinstance(variables.get("rows"), list):
        rows = variables["rows"]
        slim["table_preview"] = _trim(rows[:max_rows])
        slim["table_row_count"] = slim.get("table_row_count") or len(rows)
        slim["_note"] = "Full rows omitted; only a preview was sent to the model."

    text = json.dumps(slim, default=str, indent=2)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n...(context truncated for model window)"
    return text


def _resolve_node_provider(node: AgentNode, default_provider: AIProvider) -> AIProvider:
    """Checked at the start of every AI executor below: if this specific
    node's config explicitly names a `provider` (set by the Ask AI
    inspector's dropdowns, or a hand-authored spec — validated at spec-
    construction time by AgentNode's own model_validator, and again at
    compile time by compiler.py's _check_ai_provider_availability), build
    and use THAT provider for this call — never the spec/server default,
    and never a silent stub substitute.

    This is what makes "the selected provider/model is actually the
    provider/model used by the executor" true rather than aspirational:
    without this check, build_ai_executors() bakes in ONE provider for
    every node in the executor_overrides dict, so a per-node override in
    config would be validated but never actually honored at execution
    time. Missing credentials here raise ProviderConfigurationError
    (never caught anywhere in this codebase and silently downgraded) --
    the compile-time check already normally catches this earlier, but a
    credential can be revoked between compile and run, so this is real
    defense-in-depth, not redundant.
    """
    provider_name = node.config.get("provider") if isinstance(node.config, dict) else None
    if not provider_name:
        return default_provider

    from backend.ai_providers.registry import (
        ProviderConfigurationError,
        api_key_env_var,
        build_provider,
        has_credentials,
    )

    provider_name = str(provider_name).strip().lower()
    if not has_credentials(provider_name):
        env_var = api_key_env_var(provider_name)
        raise ProviderConfigurationError(
            f"Node '{node.id}' explicitly selects AI provider '{provider_name}', but {env_var} is "
            f"not configured on this server. Refusing to silently fall back to another provider or "
            f"a stub — set {env_var}, or remove the explicit provider selection."
        )
    model_name = node.config.get("model")
    return build_provider(provider_name, model_name)


def _system_prompt(node: AgentNode, spec: AgentSpecification, instructions: str) -> str:
    base = spec.instructions or ""
    return f"{base}\n\n{instructions}".strip()


# ---- classify -----------------------------------------------------------

def _build_classify_schema(node: AgentNode) -> type[BaseModel]:
    labels = node.config.get("labels")
    if not labels:
        raise ValueError(f"classify node '{node.id}' has no config.labels to classify into.")
    return create_model(
        f"ClassifyOutput_{node.id}",
        label=(Literal[tuple(labels)], Field(description="The single best-fitting label.")),
        confidence=(float, Field(ge=0, le=1, description="Confidence in this label, 0-1.")),
    )


def _make_classify_executor(provider: AIProvider) -> NodeExecutor:
    def execute(node: AgentNode, spec: AgentSpecification, variables: dict) -> dict:
        active_provider = _resolve_node_provider(node, provider)
        schema = _build_classify_schema(node)
        system = _system_prompt(
            node, spec,
            f"Classify the given data into exactly one of these labels: {', '.join(node.config['labels'])}.",
        )
        out = active_provider.parse(
            system=system,
            user_content=_variables_context(variables),
            schema=schema,
            max_tokens=DEFAULT_MAX_TOKENS,
        )
        return {"label": out.label, "label_confidence": out.confidence}

    return execute


# ---- extract_information ----------------------------------------------

def _build_extract_schema(node: AgentNode) -> type[BaseModel]:
    fields: list[str] = node.config.get("fields", [])
    if not fields:
        raise ValueError(f"extract_information node '{node.id}' has no config.fields to extract.")
    field_types: dict[str, str] = node.config.get("field_types", {})
    field_defs: dict[str, Any] = {}
    for f in fields:
        py_type = _FIELD_TYPE_MAP.get(field_types.get(f, "string"), str)
        # Every extracted field is Optional -- the model may legitimately
        # not find a given field in the input, and that should come back
        # as None rather than force a hallucinated value to satisfy the schema.
        field_defs[f] = (Optional[py_type], None)
    return create_model(f"ExtractOutput_{node.id}", **field_defs)


def _make_extract_executor(provider: AIProvider) -> NodeExecutor:
    def execute(node: AgentNode, spec: AgentSpecification, variables: dict) -> dict:
        active_provider = _resolve_node_provider(node, provider)
        schema = _build_extract_schema(node)
        fields = node.config.get("fields", [])
        system = _system_prompt(
            node, spec,
            f"Extract these fields from the given data: {', '.join(fields)}. "
            f"Use null for any field you cannot find -- never guess.",
        )
        out = active_provider.parse(
            system=system,
            user_content=_variables_context(variables),
            schema=schema,
            max_tokens=DEFAULT_MAX_TOKENS,
        )
        return out.model_dump()

    return execute


# ---- summarize -----------------------------------------------------------

class _SummaryOutput(BaseModel):
    summary: str


def _make_summarize_executor(provider: AIProvider) -> NodeExecutor:
    def execute(node: AgentNode, spec: AgentSpecification, variables: dict) -> dict:
        active_provider = _resolve_node_provider(node, provider)
        max_words = node.config.get("max_words")
        instruction = "Summarize the given data clearly and concisely."
        if max_words:
            instruction += f" Keep it under {max_words} words."
        system = _system_prompt(node, spec, instruction)
        out = active_provider.parse(
            system=system,
            user_content=_variables_context(variables),
            schema=_SummaryOutput,
            max_tokens=DEFAULT_MAX_TOKENS,
        )
        return {"summary": out.summary}

    return execute


# ---- make_decision ---------------------------------------------------

class _DecisionOutput(BaseModel):
    decision: str
    confidence: float = Field(ge=0, le=1)
    reasoning: str


def _make_decision_executor(provider: AIProvider) -> NodeExecutor:
    def execute(node: AgentNode, spec: AgentSpecification, variables: dict) -> dict:
        active_provider = _resolve_node_provider(node, provider)
        options = node.config.get("options")
        instruction = "Decide the best next action given the data."
        if options:
            instruction += f" Choose exactly one of: {', '.join(options)}."
        system = _system_prompt(node, spec, instruction)
        out = active_provider.parse(
            system=system,
            user_content=_variables_context(variables),
            schema=_DecisionOutput,
            max_tokens=DEFAULT_MAX_TOKENS,
        )
        return {"decision": out.decision, "decision_confidence": out.confidence, "decision_reasoning": out.reasoning}

    return execute


# ---- ai_agent / ask_ai (free-text generation, no fixed schema) ----------

def _make_freetext_executor(provider: AIProvider, default_output_key: str) -> NodeExecutor:
    def execute(node: AgentNode, spec: AgentSpecification, variables: dict) -> dict:
        active_provider = _resolve_node_provider(node, provider)
        output_key = node.config.get("output_key", default_output_key)
        prompt = node.config.get("prompt", "")
        system = _system_prompt(node, spec, "You are a node in an automated agent workflow.")
        max_chars = int(node.config.get("context_max_chars") or 12000)
        max_rows = int(node.config.get("context_max_rows") or 25)
        rag_ctx = variables.get("rag_context")
        if isinstance(rag_ctx, str) and rag_ctx.strip():
            # Lead with retrieved document text — never only index metadata.
            data_blob = (
                "Retrieved document text (rag_context):\n"
                + rag_ctx[: max_chars]
                + "\n\nOther workflow data:\n"
                + _variables_context(
                    {k: v for k, v in variables.items() if k not in ("rag_context", "rag_chunks")},
                    max_chars=max(2000, max_chars // 3),
                    max_rows=max_rows,
                )
            )
        else:
            data_blob = _variables_context(variables, max_chars=max_chars, max_rows=max_rows)
        user_content = f"{prompt}\n\n{data_blob}".strip()
        images: list[str] = []
        cfg_img = node.config.get("image_path") or node.config.get("image")
        if isinstance(cfg_img, str) and cfg_img.strip():
            images.append(cfg_img.strip())
        for key in ("rag_images", "images", "image_paths"):
            val = variables.get(key)
            if isinstance(val, list):
                images.extend(str(x) for x in val if x)
            elif isinstance(val, str) and val.strip():
                images.append(val.strip())
        single = variables.get("image_path")
        if isinstance(single, str) and single.strip():
            images.append(single.strip())
        # Index-only runs: paths listed in rag_files_indexed
        indexed = variables.get("rag_files_indexed") or []
        if isinstance(indexed, list):
            for rel in indexed:
                s = str(rel).lower()
                if s.endswith((".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")):
                    images.append(str(rel))

        import os
        from pathlib import Path as _P
        files_root = _P(os.environ.get("AGENT_FILES_DIR", "agent_platform_files")).resolve()

        def _resolve_image(p: str) -> str | None:
            path_obj = _P(p)
            candidates = []
            if path_obj.is_file():
                return str(path_obj.resolve())
            candidates.append(files_root / p)
            candidates.append(files_root / path_obj.name)
            if not path_obj.is_absolute():
                candidates.append(_P.cwd() / p)
            for c in candidates:
                try:
                    if c.is_file():
                        return str(c.resolve())
                except Exception:
                    continue
            return None

        seen: set[str] = set()
        uniq: list[str] = []
        missing: list[str] = []
        for p in images:
            resolved = _resolve_image(str(p))
            if resolved and resolved not in seen:
                seen.add(resolved)
                uniq.append(resolved)
            elif not resolved:
                missing.append(str(p))
        if missing and not uniq:
            # Fail loud — silent skip is why vision models only saw JSON metadata.
            from backend.real_executors.tools import ConnectedExecutionError
            raise ConnectedExecutionError(
                f"Ask AI could not open image(s): {missing}. "
                f"Upload into the agent files dir and use that relative path "
                f"(looked under {files_root})."
            )
        max_out = DEFAULT_MAX_TOKENS
        if node.config.get("max_tokens") is not None:
            try:
                max_out = max(64, int(node.config.get("max_tokens")))
            except (TypeError, ValueError):
                max_out = DEFAULT_MAX_TOKENS
        text = active_provider.create(
            system=system,
            user_content=user_content,
            max_tokens=max_out,
            images=uniq or None,
        )
        out = {output_key: text, "last_ai_prompt": prompt}
        # Also mirror under draft_email when ask_ai produced body-like text so Send Email finds it.
        if output_key != "draft_email" and "draft_email" not in variables:
            out.setdefault("draft_email", text)
        return out


    return execute


def build_ai_executors(provider: AIProvider | None = None) -> dict[NodeType, NodeExecutor]:
    """
    Returns a dict suitable for `compile_agent(spec, executor_overrides=...)`
    covering every AI-category node type with a real, provider-backed
    implementation. `run_agent` is deliberately NOT included here -- it
    recursively compiles+invokes a subgraph agent, which needs access to
    the AgentStore (backend/api/store.py) to resolve `subgraph.agentId`,
    so it belongs with the API layer rather than this stateless module.

    `provider=None` (the default via get_executor_overrides) resolves the
    server's configured provider via backend/ai_providers/registry.py's
    build_provider() -- AI_PROVIDER env var, defaulting to "anthropic",
    exactly preserving every existing deployment's behavior. Passing a
    provider explicitly is how tests inject a fake one (see
    backend/real_executors/test_ai_providers.py) without needing a live
    API key for either provider.
    """
    resolved_provider = provider or build_provider()
    return {
        NodeType.classify: _make_classify_executor(resolved_provider),
        NodeType.extract_information: _make_extract_executor(resolved_provider),
        NodeType.summarize: _make_summarize_executor(resolved_provider),
        NodeType.make_decision: _make_decision_executor(resolved_provider),
        NodeType.ai_agent: _make_freetext_executor(resolved_provider, default_output_key="draft_email"),
        NodeType.ask_ai: _make_freetext_executor(resolved_provider, default_output_key="ai_response"),
    }
