"""Authoritative catalogue of tools that may be bound to workflow nodes."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from backend.models.agent_spec import NodeType


@dataclass(frozen=True)
class ToolDefinition:
    id: str
    name: str
    description: str
    category: str
    node_types: tuple[NodeType, ...]
    permissions: tuple[str, ...]
    credential_required: bool
    execution: Literal["demo", "connected"]


TOOL_REGISTRY: dict[str, ToolDefinition] = {
    # execution="connected": web.search genuinely has a real, working
    # connected-mode executor (backend/real_executors/search.py, via
    # Anthropic's server-side web_search tool) -- registry metadata used
    # to claim execution="demo" (implying no real implementation exists)
    # even though real search already ran for every search_web node the
    # moment ANTHROPIC_API_KEY was configured, regardless of what a
    # binding's own `mode` said. Both problems are fixed together: the
    # executor now actually checks binding.mode (demo -> stub, connected
    # -> real search), so this registry entry can now honestly advertise
    # connected support, gated correctly by that per-binding choice.
    # credential_required=False: unlike Gmail/Sheets, connected execution
    # here needs the platform's own ANTHROPIC_API_KEY (already required
    # for any real AI functionality), never a per-binding credentialRef.
    "web.search": ToolDefinition("web.search", "Web Search", "Search the public web", "web", (NodeType.search_web,), ("read",), False, "connected"),
    "gmail.send": ToolDefinition("gmail.send", "Gmail (OAuth)", "Send email through Gmail OAuth", "communication", (NodeType.send_email,), ("send",), True, "connected"),
    "smtp.send": ToolDefinition("smtp.send", "Email (SMTP)", "Send email via SMTP (app password)", "communication", (NodeType.send_email,), ("send",), True, "connected"),
    "gmail.read": ToolDefinition("gmail.read", "Gmail (OAuth)", "Read email through Gmail OAuth", "communication", (NodeType.read_email,), ("read",), True, "connected"),
    "smtp.read": ToolDefinition("smtp.read", "Email (IMAP)", "Read email via IMAP (app password)", "communication", (NodeType.read_email,), ("read",), True, "connected"),
    "google_sheets.append_row": ToolDefinition("google_sheets.append_row", "Google Sheets", "Append a spreadsheet row", "data", (NodeType.database,), ("write",), True, "connected"),
    "local.sqlite.append_row": ToolDefinition("local.sqlite.append_row", "Local SQLite (demo)", "Append a local demo record", "data", (NodeType.database,), ("write",), False, "demo"),
    # credential_required=False: unlike Gmail/Sheets, the real executor
    # (backend/real_executors/tools.py's read_file/write_file) does local,
    # sandboxed disk I/O and genuinely never fetches a secret from the
    # credential store to do it — the "files" provider's own api_key field
    # is optional for exactly this reason (see credentials/providers.py).
    # Marking this credential_required=True used to mean selecting "File
    # reader" from the Inspector's tool dropdown alone was NOT enough:
    # compile_agent() would refuse to run until a credentialRef was set,
    # but there was (and still is) no UI to create one, so a plain
    # trigger -> read_file graph could never actually run through the app.
    "files.read": ToolDefinition("files.read", "File reader", "Read a file", "data", (NodeType.read_file,), ("read",), False, "connected"),
    "files.write": ToolDefinition("files.write", "File writer", "Write a file", "data", (NodeType.write_file,), ("write",), False, "connected"),
}


def get_tool(tool_id: str) -> ToolDefinition | None:
    return TOOL_REGISTRY.get(tool_id)


def public_registry() -> list[dict]:
    return [asdict(tool) | {"node_types": [node.value for node in tool.node_types]} for tool in TOOL_REGISTRY.values()]
