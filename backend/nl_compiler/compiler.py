"""
The NL compiler: turns a plain-language instruction into a validated,
proposed spec change.

    user message + current spec
              |
     build_system_prompt()
              |
     provider.parse_conversation(schema=MutationBatch)   <- structured outputs,
              |                                              via the SAME AIProvider
              |                                              abstraction real AI nodes use
     apply_mutations(spec, batch.mutations, expected_version=...)  <- mutation engine
              |
     MutationResult (ok + new spec, or errors)

If the model references something that doesn't exist (a typo'd node id, an
edge condition referencing a variable no upstream node produces, etc.), the
mutation engine's validation catches it — Pydantic's schema-shape guarantee
from structured outputs does NOT catch this, since "does node n_xyz exist"
depends on the current spec, not the schema. We feed that error back to the
model for a bounded number of retries rather than surfacing a raw
validation error to the end user, which is exactly the "AI -> Workflow
Reliability" problem the architecture doc flags in section 23.

Note this function is, and always was, PURE with respect to persistence:
`apply_mutations` never touches the AgentStore — it just computes what the
spec WOULD be. Whether a caller immediately persists the result (the
original /instructions route's behavior, preserved for backward
compatibility) or holds it for a person to review and explicitly apply
(the propose/apply chat routes) is entirely the caller's decision — see
backend/api/app.py's `propose_chat_instruction` / `apply_chat_proposal`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from backend.ai_providers.base import AIProvider
from backend.ai_providers.registry import build_provider
from backend.models.agent_spec import AgentSpecification, AgentSpecMutation, MutationResult
from backend.mutation.engine import apply_mutations
from backend.nl_compiler.prompts import build_system_prompt
from backend.nl_compiler.schemas import MutationBatch

def _nl_max_tokens() -> int:
    import os
    try:
        return max(1024, int(os.environ.get("NL_COMPILER_MAX_TOKENS", "8192")))
    except ValueError:
        return 8192


DEFAULT_MAX_TOKENS = _nl_max_tokens()
DEFAULT_MAX_ATTEMPTS = 3  # extra attempt for length-limit retries



class NLCompileError(Exception):
    """Raised when the model refuses, or exhausts retries without ever
    producing schema-valid structured output at all (not the same as the
    mutations failing spec validation — that's returned as MutationResult)."""


@dataclass
class NLCompileOutcome:
    reasoning: str
    result: MutationResult
    attempts: int
    # The raw mutation batch that produced `result` — echoed back to the
    # chat propose/apply routes so a client can send the SAME mutations
    # back for `apply` without needing another AI call (apply_mutations is
    # deterministic given the same mutations + spec, so re-running it
    # against a freshly-fetched spec is correct and cheap — no need for
    # server-side proposal caching).
    mutations: list[AgentSpecMutation] = field(default_factory=list)


def compile_instruction(
    spec: AgentSpecification,
    user_message: str,
    provider: AIProvider | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> NLCompileOutcome:
    """`provider=None` resolves the server's configured default provider
    via backend/ai_providers/registry.py's build_provider() — the exact
    same resolution AI-category nodes use (backend/real_executors/ai.py),
    so a deployment that sets AI_PROVIDER=openai gets the chat builder
    running on OpenAI too, with no separate configuration. Passing a
    provider explicitly is how a specific chat request can ask for a
    different provider (see backend/api/app.py's chat routes) and how
    tests inject a fake one (backend/nl_compiler/test_compiler.py)
    without needing a live API key for either provider."""
    resolved_provider = provider or build_provider()
    system_prompt = build_system_prompt(spec)
    messages: list[dict] = [{"role": "user", "content": user_message}]

    last_batch: MutationBatch | None = None
    last_result: MutationResult | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            batch = resolved_provider.parse_conversation(
                system=system_prompt,
                messages=messages,
                schema=MutationBatch,
                max_tokens=DEFAULT_MAX_TOKENS,
            )
        except Exception as exc:  # AIProviderError, or any provider-level failure
            err = str(exc).lower()
            length_hit = "length limit" in err or "max_tokens" in err or "context length" in err
            if length_hit and attempt < max_attempts:
                # Small local / 4k-context models often exhaust the window on verbose
                # MutationBatch JSON. Retry with a hard "be compact" instruction.
                messages = [
                    *messages,
                    {
                        "role": "user",
                        "content": (
                            "Previous reply hit the token length limit. "
                            "Return a MINIMAL MutationBatch: reasoning under 40 words; "
                            "at most 6 mutations; short ids; no extra commentary; "
                            "omit optional fields. Prefer: trigger_manual → read_file → ask_ai "
                            "with config.path and config.prompt only."
                        ),
                    },
                ]
                continue
            raise NLCompileError(f"The model failed to produce mutations for this instruction: {exc}") from exc

        last_batch = batch

        if not batch.mutations:
            # No change needed (e.g. the instruction was a question) — don't
            # call the mutation engine at all, so there's no spurious
            # version bump for a no-op. `batch.reasoning` IS the chat
            # response in this case — see the schema's own docstring.
            return NLCompileOutcome(
                reasoning=batch.reasoning,
                result=MutationResult(ok=True, spec=None),
                attempts=attempt,
                mutations=[],
            )

        result = apply_mutations(spec, batch.mutations, expected_version=spec.metadata.version)
        last_result = result

        if result.ok:
            return NLCompileOutcome(reasoning=batch.reasoning, result=result, attempts=attempt, mutations=batch.mutations)

        if attempt == max_attempts:
            break

        # Feed the validation failure back and ask the model to correct itself,
        # rather than surfacing a Pydantic error message straight to the user.
        error_text = "; ".join(f"{e.path}: {e.message}" for e in result.errors)
        messages.append({"role": "assistant", "content": json.dumps(batch.model_dump(mode="json"))})
        messages.append({
            "role": "user",
            "content": (
                f"Those mutations were rejected by validation: {error_text}. "
                f"Re-check the node/edge/tool ids against the current spec state above "
                f"and produce a corrected mutation batch."
            ),
        })

    assert last_batch is not None and last_result is not None
    return NLCompileOutcome(reasoning=last_batch.reasoning, result=last_result, attempts=max_attempts, mutations=last_batch.mutations)
