"""
The structured-output contract for the NL compiler.

We deliberately ask the model for a batch of AgentSpecMutations, never a
whole new spec. A whole-spec regeneration would (a) risk silently dropping
fields the model didn't think to repeat, and (b) make it impossible to
tell what actually changed for the trace/undo history. A mutation batch is
small, auditable, and — because it's fed straight into
backend/mutation/engine.py — gets exactly the same validation, atomicity,
and version-conflict handling as a canvas edit.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field

from backend.models.agent_spec import AgentSpecMutation


class MutationBatch(BaseModel):
    reasoning: str = Field(
        description=(
            "One or two plain-language sentences explaining what these mutations do. "
            "Shown directly to the user in the chat builder, so write it for them, "
            "not as an internal note."
        )
    )
    mutations: list[Annotated[AgentSpecMutation, Field(discriminator="kind")]] = Field(
        description=(
            "The ordered list of spec mutations that implement the user's instruction. "
            "Empty if the instruction requires no change (e.g. a question about the "
            "current spec)."
        )
    )
