"""
The provider abstraction AI-category node executors are built against.

Before this module, backend/real_executors/ai.py imported `anthropic.Anthropic`
directly and every executor closure called `client.messages.parse(...)` /
`client.messages.create(...)` by name -- Anthropic-specific request shapes
baked directly into the node-execution logic. That made "add OpenAI as an
option" a change to ai.py itself, not an addition alongside it.

This interface is the seam: exactly the two shapes every AI-category node
type (classify, extract_information, summarize, make_decision, ai_agent,
ask_ai) actually needs -- structured output against a dynamically-built
pydantic schema, and free-text generation -- and nothing provider-specific.
backend/real_executors/ai.py depends only on this interface now; it has no
knowledge of which concrete provider it's talking to. Adding a new
provider (Gemini, a local model server, etc.) means implementing this one
class against that provider's SDK -- no changes to ai.py, the AgentSpec
schema, the node inspector, or any node definition.

Concrete implementations: anthropic_provider.py (unchanged behavior from
before this abstraction existed) and openai_provider.py (new).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel


class AIProviderError(RuntimeError):
    """Raised when a provider genuinely fails to produce a usable result
    (an API error, a refused structured-output request, a missing
    credential resolved at call time). Never caught and silently
    downgraded to fake data anywhere in this codebase -- same "no silent
    stub fallback" principle as backend/real_executors/tools.py."""


class AIProvider(ABC):
    """Every AI-category node executor is built against this interface,
    never a concrete SDK client. `name` identifies the provider in error
    messages and the /ai/providers status endpoint -- it is NOT a secret
    and is safe to show in the frontend; the actual API key stays
    server-side inside the concrete implementation's constructor,
    resolved from that provider's own environment variable
    (backend/ai_providers/registry.py), never accepted from a request
    body or an AgentSpec field.
    """

    name: str

    @abstractmethod
    def parse(self, *, system: str, user_content: str, schema: type[BaseModel], max_tokens: int) -> BaseModel:
        """Structured output: ask the model to fill in `schema` from
        `user_content` (with `system` as the system prompt), returning a
        validated instance of it. Used by classify, extract_information,
        summarize, and make_decision -- every AI node type with a fixed
        output shape."""

    @abstractmethod
    def parse_conversation(
        self, *, system: str, messages: list[dict[str, str]], schema: type[BaseModel], max_tokens: int
    ) -> BaseModel:
        """Multi-turn counterpart to `parse()`: `messages` is a list of
        `{"role": "user"|"assistant", "content": str}` turns (the same
        shape every provider SDK already uses for a message history).
        Used by the NL compiler (backend/nl_compiler/compiler.py), which
        needs to feed a validation failure back to the model and ask for
        a corrected mutation batch -- a genuine back-and-forth, not a
        single request/response, so it can't be expressed through
        `parse()`'s single `user_content` string."""

    @abstractmethod
    def create(self, *, system: str, user_content: str, max_tokens: int, images: list[str] | None = None) -> str:
        """Free-text generation: no fixed schema, just the model's raw
        text response. Used by ai_agent and ask_ai."""
