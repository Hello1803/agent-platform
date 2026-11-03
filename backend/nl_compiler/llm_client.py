"""
The NL compiler talks to "a model", not to a specific vendor's SDK — this
keeps the promise in architecture doc section 21 ("the platform should
avoid coupling its public architecture to a single LLM provider").

Two implementations are provided:
  - AnthropicLLMClient: a real implementation for production use.
  - FakeLLMClient: a deterministic, scripted client used by
    backend/run_nl_compiler.py so the compiler pipeline can be demonstrated
    and tested without a live API key.
"""

from __future__ import annotations

from typing import Protocol


class LLMClient(Protocol):
    def complete(self, system: str, user: str) -> str:
        """Return the model's raw text response to a single-turn request."""
        ...


class AnthropicLLMClient:
    """
    Thin wrapper around the Anthropic SDK. Requires `pip install anthropic`
    and an ANTHROPIC_API_KEY in the environment.
    """

    def __init__(self, model: str = "claude-sonnet-5", max_tokens: int = 4096):
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "AnthropicLLMClient requires the 'anthropic' package: pip install anthropic"
            ) from e
        self._client = anthropic.Anthropic()
        self._model = model
        self._max_tokens = max_tokens

    def complete(self, system: str, user: str) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in response.content if block.type == "text")


class FakeLLMClient:
    """
    A scripted client for tests/demos: takes a list of canned responses and
    returns them in order, one per call. Lets run_nl_compiler.py exercise
    the full parse -> validate -> (self-repair retry) -> apply pipeline
    deterministically, including a response that's deliberately broken JSON
    to demonstrate the self-repair loop.
    """

    def __init__(self, scripted_responses: list[str]):
        self._responses = list(scripted_responses)
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if not self._responses:
            raise RuntimeError("FakeLLMClient ran out of scripted responses.")
        return self._responses.pop(0)
