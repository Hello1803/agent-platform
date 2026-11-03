"""
Real `search_web` executor, using Anthropic's server-side web search tool
(`web_search_20250305`) rather than a canned stub.

Why this and not a dedicated search API: this sandbox's outbound egress is
allowlisted to a handful of domains (see network_configuration) that does
NOT include any search provider (Bing, Serper, Google Custom Search,
etc.) -- but it DOES include api.anthropic.com, which every AI node here
already depends on. Anthropic's Messages API supports a server-side web
search tool: the search itself happens on Anthropic's infrastructure, not
this process's, so it needs no search-provider credential or extra
egress at all -- just the ANTHROPIC_API_KEY every real AI executor in this
repo already requires. That makes it the one "real, not stub" web search
implementation actually achievable here, the same reasoning that made
SQLite the first real `database` executor and local-disk I/O the first
real `read_file` one: pick the real integration this environment can
actually prove works end-to-end, not the one that would need
infrastructure nobody's given this process access to.

Registry note: web.search stays registered as execution="demo"
(backend/tools/registry.py) rather than gaining a "connected" mode -- it
needs no separate credential (see above), so there's no meaningful
distinction between "demo" and "connected" for it the way there is for
Gmail. What changes here is what the DEMO executor itself does: same
precedent as `database`, whose demo-mode executor writes to real SQLite
instead of an in-memory stub.
"""
from __future__ import annotations

from typing import Any

from anthropic import Anthropic

from backend.compiler.executors import NodeExecutor, exec_search_web
from backend.models.agent_spec import AgentNode, AgentSpecification, ToolBinding

DEFAULT_MODEL = "claude-sonnet-5"
DEFAULT_MAX_TOKENS = 1024


def _resolve_query(node: AgentNode, spec: AgentSpecification, variables: dict[str, Any]) -> str | None:
    """Query resolution order: an explicit static node.config.query first
    (the common case for a fixed research topic), then a `query` variable
    an earlier node populated (e.g. an extract_information step pulling a
    topic out of user input), then the agent's own stated goal as a last,
    coarse resort -- better than failing outright on a spec that never
    bothered to wire a query explicitly."""
    query = node.config.get("query") if node.config else None
    if query:
        return str(query)
    if variables.get("query"):
        return str(variables["query"])
    if spec.goal:
        return spec.goal
    return None


def _extract_result(response: Any) -> dict:
    text_parts: list[str] = []
    sources: list[dict[str, str]] = []
    seen_urls: set[str] = set()

    for block in response.content:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            text_parts.append(block.text)
        elif block_type == "web_search_tool_result":
            content = getattr(block, "content", None) or []
            for result in content:
                url = getattr(result, "url", None)
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    sources.append({"title": getattr(result, "title", "") or "", "url": url})

    return {
        "search_results": "".join(text_parts).strip(),
        "search_sources": sources,
    }


def _real_search_web(node: AgentNode, spec: AgentSpecification, variables: dict, client: Anthropic, model: str) -> dict:
    query = _resolve_query(node, spec, variables)
    if not query:
        raise ValueError(
            f"search_web node '{node.id}' has no query -- set node.config.query, populate a "
            "'query' variable from an earlier node, or give the agent a goal to fall back to."
        )

    response = client.messages.create(
        model=model,
        max_tokens=DEFAULT_MAX_TOKENS,
        tools=[{"type": "web_search_20250305", "name": "web_search"}],
        messages=[
            {
                "role": "user",
                "content": (
                    f"Search the web for: {query}\n\n"
                    "Give a concise, factual answer (2-5 sentences) synthesizing what you find."
                ),
            }
        ],
    )
    return _extract_result(response)


def _resolve_search_binding(node: AgentNode, spec: AgentSpecification) -> ToolBinding | None:
    """Same lookup shape as real_executors/tools.py's `_resolve_binding` --
    finds the tool binding actually granting this node's 'read' capability,
    so search_web's real executor can check its `.mode` exactly like every
    other real tool executor in this codebase does. Returns None if
    somehow uncompiled/no binding (shouldn't happen once compiler.py's
    REQUIRED_CAPABILITY check has run, but this function has no compiler
    guarantee of its own callers)."""
    if not node.tool_refs:
        return None
    bindings_by_id = {t.id: t for t in spec.tools}
    for ref in node.tool_refs:
        binding = bindings_by_id.get(ref)
        if binding and "read" in binding.permissions:
            return binding
    return None


def build_search_web_executor(client: Anthropic | None = None, model: str = DEFAULT_MODEL) -> NodeExecutor:
    """Mode-aware, matching every other real tool executor in this
    codebase (real_executors/tools.py's send_email/read_email/database/
    etc.): a search_web node's TOOL BINDING controls whether it gets real
    search or the demo stub, via `mode: "demo"` vs `mode: "connected"` --
    NOT ambient server configuration alone. Before this fix, real search
    ran for EVERY search_web node the instant ANTHROPIC_API_KEY existed
    server-side, completely ignoring a binding explicitly set to
    mode="demo" -- exactly the "explicit tool binding is not preserved"
    bug this closes.

    `client=None` (the default in production, via get_executor_overrides)
    means 'use a real Anthropic client for connected-mode bindings.'
    Passing a client explicitly is how tests inject a fake one instead of
    needing a live ANTHROPIC_API_KEY and live egress (see
    backend/real_executors/test_search.py) -- same pattern
    real_executors/tools.py uses (injectable httpx transport) for
    Gmail/Sheets.
    """
    real_client = client or Anthropic()

    def execute(node: AgentNode, spec: AgentSpecification, variables: dict[str, Any]) -> dict:
        binding = _resolve_search_binding(node, spec)
        if binding is not None and binding.mode == "demo":
            return exec_search_web(node, spec, variables)
        return _real_search_web(node, spec, variables, real_client, model)

    return execute


def build_search_web_executor_with_fallback(client: Anthropic | None) -> NodeExecutor:
    """Used by get_executor_overrides() when no ANTHROPIC_API_KEY is
    configured AT ALL on this server: every search_web node (demo or
    connected binding alike) falls back to the plain stub, since a
    connected-mode binding genuinely can't run without it anyway. This is
    a SERVER-LEVEL absence of the credential, not a per-binding choice --
    distinct from build_search_web_executor's per-binding mode check
    above, which still applies once a real client IS available."""
    if client is None:
        def execute(node: AgentNode, spec: AgentSpecification, variables: dict[str, Any]) -> dict:
            return exec_search_web(node, spec, variables)

        return execute
    return build_search_web_executor(client)
