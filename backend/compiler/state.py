"""
The LangGraph state schema every compiled AgentSpecification shares.

Kept deliberately generic: `variables` is the data bag nodes read/write
(what the doc calls "state" informally — extracted fields, flags like
`approved`, tool outputs), and `trace` is the append-only execution log
that powers the Execution Trace UI (architecture doc section 12).
"""

from __future__ import annotations

import operator
from datetime import datetime, timezone
from typing import Annotated, Any, Literal
from typing_extensions import TypedDict


def _merge_variables(current: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    return {**current, **update}


class TraceEvent(TypedDict):
    node_id: str
    node_label: str
    status: Literal["success", "failed", "waiting"]
    detail: str
    timestamp: str


class AgentState(TypedDict):
    variables: Annotated[dict[str, Any], _merge_variables]
    trace: Annotated[list[TraceEvent], operator.add]


def make_trace_event(node_id: str, node_label: str, status: str, detail: str) -> TraceEvent:
    return TraceEvent(
        node_id=node_id,
        node_label=node_label,
        status=status,  # type: ignore[typeddict-item]
        detail=detail,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def initial_state(seed_variables: dict[str, Any] | None = None) -> AgentState:
    return AgentState(variables=dict(seed_variables or {}), trace=[])
