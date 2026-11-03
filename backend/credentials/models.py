"""Request/response and internal record shapes for the credential store.

Two shapes matter here and they must never merge into one:

- `Credential` — the internal record, holding the encrypted secret bytes.
  Only ever constructed/read inside `backend/credentials/store.py` and the
  real executors that need the decrypted value at call time.
- `CredentialSummary` — the only shape that's allowed to cross the API
  boundary. It carries everything a UI needs to show a connections list
  (provider, name, status, timestamps) and *nothing* that could
  reconstruct the secret. `AgentSpecification.toolRefs[].credentialRef`
  only ever stores this record's `id` — never a token, key, or password —
  so a spec dumped to the frontend, logged, or handed to the NL compiler's
  LLM prompt carries no secret material by construction.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


CredentialStatus = Literal["unvalidated", "valid", "invalid"]


class CredentialCreateRequest(BaseModel):
    provider: str = Field(description="Provider id from the credential provider registry, e.g. 'gmail'.")
    name: str = Field(description="Human-readable label, e.g. 'Support inbox'.")
    secret: dict = Field(description="Provider-shaped secret payload. Never echoed back by the API.")
    owner_id: str | None = Field(
        default=None,
        description="Owning user/org id. Defaults to the request's authenticated principal once auth exists (P0.4); "
        "'default' in the interim single-tenant dev setup.",
    )


class CredentialSummary(BaseModel):
    """Public shape. No secret material, ever."""

    id: str
    provider: str
    name: str
    owner_id: str
    status: CredentialStatus
    created_at: str
    updated_at: str
    last_validated_at: str | None = None
    last_validation_error: str | None = None


class Credential(BaseModel):
    """Internal record. Never returned from an API route."""

    id: str
    provider: str
    name: str
    owner_id: str
    status: CredentialStatus = "unvalidated"
    encrypted_secret: bytes
    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)
    last_validated_at: str | None = None
    last_validation_error: str | None = None

    def summary(self) -> CredentialSummary:
        return CredentialSummary(
            id=self.id,
            provider=self.provider,
            name=self.name,
            owner_id=self.owner_id,
            status=self.status,
            created_at=self.created_at,
            updated_at=self.updated_at,
            last_validated_at=self.last_validated_at,
            last_validation_error=self.last_validation_error,
        )
