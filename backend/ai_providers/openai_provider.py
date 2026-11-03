"""OpenAI implementation of the AIProvider interface -- the first
alternative to Anthropic, proving the abstraction in base.py actually
decouples node execution from a specific provider's SDK.

Structured output uses the OpenAI SDK's `chat.completions.parse()` with a
pydantic model as `response_format` (native support since openai>=1.40,
verified against openai==3.8.0 in this environment) -- the direct
equivalent of Anthropic's `messages.parse(output_format=schema)`.
"""
from __future__ import annotations

from openai import OpenAI
from pydantic import BaseModel

from backend.ai_providers.base import AIProvider, AIProviderError

DEFAULT_MODEL = "gpt-4o"


class OpenAIProvider(AIProvider):
    name = "openai"

    def __init__(self, client: OpenAI | None = None, model: str = DEFAULT_MODEL):
        # client=None means "construct a real OpenAI() client", which
        # reads OPENAI_API_KEY from the environment itself -- same
        # server-side-only credential handling as AnthropicProvider.
        self._client = client or OpenAI()
        self._model = model

    def parse(self, *, system: str, user_content: str, schema: type[BaseModel], max_tokens: int) -> BaseModel:
        return self.parse_conversation(
            system=system, messages=[{"role": "user", "content": user_content}], schema=schema, max_tokens=max_tokens
        )

    def parse_conversation(
        self, *, system: str, messages: list[dict[str, str]], schema: type[BaseModel], max_tokens: int
    ) -> BaseModel:
        try:
            completion = self._client.chat.completions.parse(
                model=self._model,
                max_completion_tokens=max_tokens,
                messages=[{"role": "system", "content": system}, *messages],
                response_format=schema,
            )
        except Exception as exc:  # openai raises several distinct exception types on API failures
            raise AIProviderError(f"OpenAI structured-output request failed: {exc}") from exc

        choice = completion.choices[0]
        if choice.message.parsed is None:
            reason = choice.message.refusal or getattr(choice, "finish_reason", "unknown reason")
            raise AIProviderError(f"OpenAI did not return structured output ({reason}).")
        return choice.message.parsed

    def create(self, *, system: str, user_content: str, max_tokens: int, images: list[str] | None = None) -> str:
        """OpenAI-compatible chat. `images` is a list of local file paths or data-URLs."""
        import base64
        from pathlib import Path as _P

        user_parts: list = [{"type": "text", "text": user_content}]
        loaded = 0
        for img in images or []:
            if not img:
                continue
            if str(img).startswith("data:"):
                user_parts.append({"type": "image_url", "image_url": {"url": str(img)}})
                loaded += 1
                continue
            p = _P(img)
            if not p.is_file():
                continue
            suffix = p.suffix.lower().lstrip(".") or "png"
            mime = {
                "png": "image/png",
                "jpg": "image/jpeg",
                "jpeg": "image/jpeg",
                "gif": "image/gif",
                "webp": "image/webp",
            }.get(suffix, "image/png")
            b64 = base64.b64encode(p.read_bytes()).decode("ascii")
            user_parts.append(
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}
            )
            loaded += 1

        if images and loaded == 0:
            raise AIProviderError(
                f"Vision request listed {len(images)} image(s) but none could be read as files: {images}"
            )

        # Multimodal only when we actually attached an image; pure text stays a string
        # for maximum compatibility with non-vision OpenAI-compatible servers.
        user_msg_content = user_parts if loaded > 0 else user_content

        try:
            completion = self._client.chat.completions.create(
                model=self._model,
                max_completion_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_msg_content},
                ],
            )
        except Exception as exc:
            raise AIProviderError(f"OpenAI request failed: {exc}") from exc
        return completion.choices[0].message.content or ""
