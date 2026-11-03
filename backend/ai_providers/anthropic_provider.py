"""Anthropic implementation of the AIProvider interface.

Behavior is unchanged from before the provider abstraction existed --
this is the same `client.messages.parse`/`client.messages.create` calls
that used to live directly inside backend/real_executors/ai.py, just
moved behind the AIProvider interface so ai.py no longer needs to know
it's talking to Anthropic specifically.
"""
from __future__ import annotations

from anthropic import Anthropic
from pydantic import BaseModel

from backend.ai_providers.base import AIProvider, AIProviderError

DEFAULT_MODEL = "claude-sonnet-5"


class AnthropicProvider(AIProvider):
    name = "anthropic"

    def __init__(self, client: Anthropic | None = None, model: str = DEFAULT_MODEL):
        # client=None means "construct a real Anthropic() client", which
        # reads ANTHROPIC_API_KEY from the environment itself -- the key
        # never passes through this constructor's arguments, let alone
        # anything reachable from a request body.
        self._client = client or Anthropic()
        self._model = model

    def parse(self, *, system: str, user_content: str, schema: type[BaseModel], max_tokens: int) -> BaseModel:
        return self.parse_conversation(
            system=system, messages=[{"role": "user", "content": user_content}], schema=schema, max_tokens=max_tokens
        )

    def parse_conversation(
        self, *, system: str, messages: list[dict[str, str]], schema: type[BaseModel], max_tokens: int
    ) -> BaseModel:
        try:
            response = self._client.messages.parse(
                model=self._model,
                max_tokens=max_tokens,
                system=system,
                messages=messages,
                output_format=schema,
            )
        except Exception as exc:  # anthropic raises several distinct exception types on API failures
            raise AIProviderError(f"Anthropic structured-output request failed: {exc}") from exc
        if getattr(response, "stop_reason", None) == "refusal":
            raise AIProviderError("Anthropic declined to produce structured output for this request.")
        return response.parsed_output

    def create(self, *, system: str, user_content: str, max_tokens: int, images: list[str] | None = None) -> str:
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user_content}],
            )
        except Exception as exc:
            raise AIProviderError(f"Anthropic request failed: {exc}") from exc
        return "".join(block.text for block in response.content if block.type == "text")
