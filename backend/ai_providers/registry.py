"""Server-side AI provider selection and credential resolution.

**Provider selection and credentials are resolved ENTIRELY here, server-side.**
Nothing about which provider is active, or any API key, ever crosses into
an AgentSpec, an API response, or anything the frontend can read:

  - Which provider is active is chosen by the `AI_PROVIDER` environment
    variable on the server (default: "anthropic", preserving every
    existing deployment's behavior unchanged). A spec MAY optionally name
    a provider via `runtimeConfig.model` (format "<provider>:<model>",
    e.g. "openai:gpt-4o") -- see backend/models/agent_spec.py's
    RuntimeConfig validator -- but that field only ever holds a provider
    *name* and model *name*, both public, non-secret strings; it can
    never hold or influence which credential is used, since credential
    resolution (the two lines below, `Anthropic()` / `OpenAI()`) reads
    straight from environment variables this process controls, not from
    anything in the spec or request.
  - Each provider's API key comes from its own conventional environment
    variable (ANTHROPIC_API_KEY, OPENAI_API_KEY) via that provider SDK's
    own client constructor -- this module never even reads the key value
    itself, just checks whether the variable is *present* so it can
    decide whether to fall back to demo/stub execution.

Adding Gemini (or any other provider) means: implement a GeminiProvider
against backend/ai_providers/base.py's interface, add one line to
PROVIDER_FACTORIES below, and add its own environment variable to
`ENV_VAR_BY_PROVIDER`. Nothing else in this file, ai.py, the schema, or
the inspector needs to know a new provider exists ahead of time --
SUPPORTED_PROVIDERS is what the inspector/validation actually check
against.
"""
from __future__ import annotations

import os

from backend.ai_providers.base import AIProvider

# Providers with a real, working implementation right now. Referenced by
# the inspector (packages/canvas) and by RuntimeConfig.model's validator
# (backend/models/agent_spec.py) so a user can't configure a provider
# name that silently does nothing at run time.
SUPPORTED_PROVIDERS: tuple[str, ...] = ("anthropic", "openai", "lmstudio")

# Named, but not yet implemented -- kept separate from SUPPORTED_PROVIDERS
# so validation can give a specific "not yet available" message instead of
# a generic "unknown provider" one. Add a provider here the moment you
# start implementing it, and move it up to SUPPORTED_PROVIDERS the moment
# a real AIProvider subclass exists for it.
PLANNED_PROVIDERS: tuple[str, ...] = ("gemini",)

ENV_VAR_BY_PROVIDER: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "lmstudio": "LM_STUDIO_BASE_URL",
}

DEFAULT_PROVIDER = "anthropic"


class UnsupportedProviderError(ValueError):
    pass


class ModelValidationError(ValueError):
    """A model name doesn't belong to the provider it's paired with (e.g.
    an Anthropic model under 'openai:', or vice versa) -- raised by
    validate_provider_model(), used both server-side (RuntimeConfig.model
    and AgentNode.config validators in backend/models/agent_spec.py) and
    mirrored client-side (specToFlow.ts's isValidProviderModel) so a
    mismatched pair is rejected the same way in both places."""


class ProviderConfigurationError(RuntimeError):
    """Raised when a spec/node EXPLICITLY selected a provider (as opposed
    to silently falling back to the server default) and that provider's
    credentials aren't configured. This is deliberately NOT caught and
    downgraded to stub execution anywhere -- see get_executor_overrides'
    and _apply_spec_provider_override's docstrings in backend/api/app.py
    for exactly where "explicit selection" is detected and this is
    raised instead of silently substituting another provider or a stub."""


# Curated, non-exhaustive lists of known-good model names per provider --
# used to populate the /ai/providers response (so the frontend's model
# dropdown doesn't hardcode this) and as a first line of validation. The
# authoritative validation is the prefix check in validate_provider_model
# below (so a legitimate newer model not yet added here still passes);
# this list exists for a good default and a clean dropdown, not as the
# sole gate.
KNOWN_MODELS: dict[str, list[dict[str, str]]] = {
    "anthropic": [
        {"id": "claude-sonnet-5", "label": "Claude Sonnet 5"},
        {"id": "claude-opus-5", "label": "Claude Opus 5"},
        {"id": "claude-haiku-4-5-20251001", "label": "Claude Haiku 4.5"},
    ],
    "openai": [
        {"id": "gpt-4o", "label": "GPT-4o"},
        {"id": "gpt-4o-mini", "label": "GPT-4o mini"},
        {"id": "o3-mini", "label": "o3-mini"},
    ],
    "lmstudio": [
        {"id": "local-model", "label": "Local model (whatever is loaded in LM Studio)"},
    ],
}

# A model name unambiguously belongs to a provider if it starts with one
# of these prefixes -- used to reject an impossible pairing (e.g.
# "openai:claude-sonnet-5") even for a model not in KNOWN_MODELS above.
_MODEL_PREFIXES: dict[str, tuple[str, ...]] = {
    "anthropic": ("claude-",),
    "openai": ("gpt-", "o1-", "o1", "o3-", "o3", "o4-", "chatgpt-"),
    "lmstudio": (),
}


def default_model_for(provider_name: str) -> str:
    models = KNOWN_MODELS.get(provider_name)
    if not models:
        raise UnsupportedProviderError(f"No known models for provider '{provider_name}'.")
    return models[0]["id"]


def validate_provider_model(provider_name: str, model_name: str) -> None:
    """Raises ModelValidationError on unambiguous cross-provider model mismatch.
    lmstudio is exempt (local model ids are arbitrary)."""
    if provider_name == "lmstudio":
        return
    prefixes = _MODEL_PREFIXES.get(provider_name, ())
    for other_provider, other_prefixes in _MODEL_PREFIXES.items():
        if other_provider == provider_name or other_provider == "lmstudio":
            continue
        if any(model_name.startswith(p) for p in other_prefixes) and not any(
            model_name.startswith(p) for p in prefixes
        ):
            raise ModelValidationError(
                f"Model '{model_name}' belongs to provider '{other_provider}', not '{provider_name}'."
            )


def api_key_env_var(provider_name: str) -> str:
    try:
        return ENV_VAR_BY_PROVIDER[provider_name]
    except KeyError:
        raise UnsupportedProviderError(f"Unknown AI provider '{provider_name}'.") from None


def has_credentials(provider_name: str) -> bool:
    """Cloud providers need API keys. lmstudio is always selectable (local server)."""
    if provider_name == "lmstudio":
        return True
    return bool(os.environ.get(api_key_env_var(provider_name)))


def resolve_provider_name(explicit: str | None = None) -> str:
    """Resolution order: an explicit name (e.g. from a parsed
    `runtimeConfig.model`) wins; otherwise the server's AI_PROVIDER
    environment variable; otherwise the default. This is the ONE place
    that decides "which provider" -- get_executor_overrides and
    real_executors/search.py both go through it rather than each
    hardcoding their own fallback."""
    name = (explicit or os.environ.get("AI_PROVIDER") or DEFAULT_PROVIDER).strip().lower()
    if name in PLANNED_PROVIDERS:
        raise UnsupportedProviderError(
            f"AI provider '{name}' is planned but not implemented yet. "
            f"Currently supported: {', '.join(SUPPORTED_PROVIDERS)}."
        )
    if name not in SUPPORTED_PROVIDERS:
        raise UnsupportedProviderError(
            f"Unknown AI provider '{name}'. Supported: {', '.join(SUPPORTED_PROVIDERS)}."
        )
    return name


def build_provider(name: str | None = None, model: str | None = None) -> AIProvider:
    """Construct the resolved provider client."""
    resolved = resolve_provider_name(name)
    if resolved == "anthropic":
        from backend.ai_providers.anthropic_provider import DEFAULT_MODEL, AnthropicProvider
        return AnthropicProvider(model=model or DEFAULT_MODEL)
    if resolved == "openai":
        from backend.ai_providers.openai_provider import DEFAULT_MODEL, OpenAIProvider
        return OpenAIProvider(model=model or DEFAULT_MODEL)
    if resolved == "lmstudio":
        from backend.ai_providers.openai_provider import OpenAIProvider
        from openai import OpenAI
        base_url = (os.environ.get("LM_STUDIO_BASE_URL") or "http://127.0.0.1:1234/v1").rstrip("/")
        client = OpenAI(base_url=base_url, api_key=os.environ.get("LM_STUDIO_API_KEY") or "lm-studio")
        return OpenAIProvider(client=client, model=model or "local-model")
    raise UnsupportedProviderError(f"No provider implementation registered for '{resolved}'.")


def parse_model_string(value: str | None) -> tuple[str | None, str | None]:
    """Parses RuntimeConfig.model's "<provider>:<model>" convention (see
    backend/models/agent_spec.py). A bare model name with no colon is
    treated as a model for whichever provider ends up resolved elsewhere
    (returns provider=None). Returns (provider, model), either of which
    may be None."""
    if not value:
        return None, None
    if ":" in value:
        provider, _, model = value.partition(":")
        return provider.strip().lower() or None, model.strip() or None
    return None, value.strip() or None
