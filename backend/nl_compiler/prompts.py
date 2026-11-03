"""
Builds the system prompt for the NL compiler: the node-type catalog (so the
model never invents a node type outside what the platform actually
supports) plus a summary of the CURRENT spec state (so instructions like
"ask me before sending the email" can be resolved against the agent that
actually exists, not a generic template).
"""

from __future__ import annotations

from backend.models.agent_spec import AgentSpecification, NodeCategory, NodeType

_NODE_TYPES_BY_CATEGORY: dict[NodeCategory, list[NodeType]] = {
    NodeCategory.ai: [
        NodeType.ai_agent, NodeType.ask_ai, NodeType.classify, NodeType.extract_information,
        NodeType.summarize, NodeType.make_decision, NodeType.run_agent,
    ],
    NodeCategory.tool: [
        NodeType.search_web, NodeType.open_website, NodeType.read_email, NodeType.send_email,
        NodeType.search_documents, NodeType.read_file, NodeType.write_file, NodeType.database,
        NodeType.api_call, NodeType.code, NodeType.preview, NodeType.rag_index, NodeType.rag_query, NodeType.browser,
    ],
    NodeCategory.control: [
        NodeType.if_else, NodeType.switch, NodeType.loop, NodeType.parallel,
        NodeType.merge, NodeType.wait, NodeType.retry,
    ],
    NodeCategory.human: [
        NodeType.human_approval, NodeType.ask_user, NodeType.review, NodeType.confirm,
    ],
    NodeCategory.trigger: [
        NodeType.trigger_manual, NodeType.trigger_schedule, NodeType.trigger_email,
        NodeType.trigger_webhook, NodeType.trigger_file_change, NodeType.trigger_external_event,
    ],
}


def _node_catalog_block() -> str:
    lines = []
    for category, types in _NODE_TYPES_BY_CATEGORY.items():
        lines.append(f"{category.value}: " + ", ".join(t.value for t in types))
    return "\n".join(lines)


def _spec_summary_block(spec: AgentSpecification) -> str:
    nodes = "\n".join(
        f"  - {n.id} [{n.type.value}, {n.category.value}, {n.execution_type.value}]: \"{n.label}\""
        for n in spec.nodes
    ) or "  (no nodes yet)"

    edges = "\n".join(
        f"  - {e.id}: {e.source} -> {e.target}"
        + (f"  [when {e.condition.expression}]" if e.condition and e.condition.expression else "")
        for e in spec.edges
    ) or "  (no edges yet)"

    tools = "\n".join(
        f"  - {t.id}: {t.tool_id} (permissions: {', '.join(t.permissions) or 'none'})"
        for t in spec.tools
    ) or "  (no tool bindings yet)"

    approvals = "\n".join(
        f"  - {r.id}: {r.applies_to.action_category or r.applies_to.node_id} -> {r.autonomy_level.value}"
        for r in spec.human_approval
    ) or "  (no approval rules yet)"

    return (
        f"Agent name: {spec.metadata.name}\n"
        f"Goal: {spec.goal}\n"
        f"Current spec version: {spec.metadata.version}\n\n"
        f"Nodes:\n{nodes}\n\n"
        f"Edges:\n{edges}\n\n"
        f"Tool bindings:\n{tools}\n\n"
        f"Approval rules:\n{approvals}"
    )


def build_system_prompt(spec: AgentSpecification) -> str:
    return f"""You are the NL compiler for a no-code agent platform. Output a MutationBatch (structured) only.

Node types by category:
{_node_catalog_block()}

Mutation kinds: add_node, update_node, remove_node, add_edge, update_edge, remove_edge, update_metadata, add_tool, update_tool, remove_tool, add_approval_rule, update_goal. Prefer the smallest mutation set.

Rules:
- Reuse existing ids from the current spec; new nodes use n_*, edges e_*.
- Edges: sourceHandle="output", targetHandle="input" only. Edges set run order; data is a shared variable bag (not named ports).
- executionType "probabilistic" ONLY for: ai_agent, ask_ai, classify, extract_information, summarize, make_decision, run_agent. All other nodes: "deterministic".
- read_file/write_file: set config.path (e.g. "sample.csv"). No tool binding required — never set toolRefs on these nodes. write_file after ask_ai: content_variable "ai_response".
- if_else/switch: node is passthrough; put conditions on EDGES via condition: {{"expression": "amount > 1000", "kind": "deterministic"}}.
- Keep reasoning under 60 words. Prefer 2-6 mutations for simple asks (e.g. read file + ask_ai).
- Sensitive actions: prefer human_approval unless user opts out.
- Questions only: empty mutations + short reasoning.

Current spec:
{_spec_summary_block(spec)}
"""

