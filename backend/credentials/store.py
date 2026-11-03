"""Server-side credential store.

This is the concrete fix for the P0 gap this repo's README calls out
explicitly: nothing before this module stored a secret anywhere — a
`ToolBinding.credential_ref` was just a string nobody checked existed, and
`real_executors/` had no way to obtain a real Gmail/Sheets/Slack secret
even if it wanted to. This module is the missing other half.

Two properties matter, and both are enforced structurally rather than by
convention:

1. **Secrets never leave the process encrypted-at-rest, and never appear in
   an API response.** `Credential.encrypted_secret` is Fernet-encrypted
   bytes; `CredentialStore.get_secret()` — the only method that returns a
   decrypted dict — is intentionally not exposed on any API route (see
   `backend/api/app.py`'s `/credentials` routes returning `CredentialSummary`
   only). Only `real_executors/` calls `get_secret()`, at the moment a
   node actually needs to make the call, never earlier and never cached
   into `AgentState`/`variables` where a trace or a chat transcript could
   leak it.

2. **A spec can only reference a credential that actually exists and
   actually matches the tool's required provider.** That check used to be
   "is `credential_ref` non-empty" (`backend/compiler/compiler.py`); this
   module is what lets that check become real (see
   `validate_credential_ref` below, wired into the compiler in the same
   changeset).

Encryption key handling: `CREDENTIAL_ENCRYPTION_KEY` (a urlsafe-base64
Fernet key) should be set in any real deployment and must be durable
across restarts — losing it makes every stored secret permanently
undecryptable, by design (there is no back door). If it's unset, a key is
generated for this process only, with a loud one-time warning (the same
"not a silent fallback" pattern `get_executor_overrides` already uses for
a missing `ANTHROPIC_API_KEY`) — fine for local dev, actively wrong for
anything that needs to survive a restart, which is exactly why it warns
instead of silently doing the convenient thing.
"""
from __future__ import annotations

import os
import sys
import threading
import uuid
from dataclasses import dataclass

from cryptography.fernet import Fernet, InvalidToken

from backend.credentials.models import Credential, CredentialSummary
from backend.credentials.providers import (
    CredentialValidationError,
    provider_for_tool,
    validate_secret,
)


class CredentialNotFoundError(KeyError):
    pass


class CredentialAccessError(PermissionError):
    """Raised when a caller tries to touch a credential it doesn't own."""


def _resolve_fernet() -> Fernet:
    key = os.environ.get("CREDENTIAL_ENCRYPTION_KEY")
    if key:
        return Fernet(key.encode() if isinstance(key, str) else key)
    generated = Fernet.generate_key()
    print(
        "[credentials] CREDENTIAL_ENCRYPTION_KEY is not set — using a "
        "process-local key generated just now. Every credential stored "
        "under this key is permanently unreadable after this process "
        "exits. Set CREDENTIAL_ENCRYPTION_KEY to a durable value "
        "(Fernet.generate_key()) before storing anything that needs to "
        "survive a restart.",
        file=sys.stderr,
    )
    return Fernet(generated)


@dataclass
class ConnectionValidationResult:
    ok: bool
    error: str | None = None


class CredentialStore:
    """In-memory credential store, encrypted at rest.

    Same "dict behind a lock" shape as `backend.api.store.AgentStore` —
    deliberately the simplest thing that satisfies the contract (encrypted
    secret, owner-scoped access, existence/provider validation) before
    committing to where these actually get persisted. A Postgres-backed
    implementation only needs to persist `id/provider/name/owner_id/status/
    encrypted_secret/timestamps` as columns; the Fernet-encrypted blob is
    already opaque so it needs no special column type.
    """

    def __init__(self, fernet: Fernet | None = None) -> None:
        self._lock = threading.Lock()
        self._credentials: dict[str, Credential] = {}
        self._fernet = fernet or _resolve_fernet()

    def create(self, *, provider: str, name: str, secret: dict, owner_id: str) -> CredentialSummary:
        # Format/shape validation happens before anything is stored — a
        # malformed credential never makes it into the store at all.
        validate_secret(provider, secret)
        encrypted = self._fernet.encrypt(_serialize(secret))
        record = Credential(
            id=f"cred_{uuid.uuid4().hex[:12]}",
            provider=provider,
            name=name,
            owner_id=owner_id,
            encrypted_secret=encrypted,
        )
        with self._lock:
            self._credentials[record.id] = record
        return record.summary()

    def list(self, owner_id: str) -> list[CredentialSummary]:
        with self._lock:
            return [c.summary() for c in self._credentials.values() if c.owner_id == owner_id]

    def get_summary(self, credential_id: str, owner_id: str) -> CredentialSummary:
        return self._get(credential_id, owner_id).summary()

    def get_secret(self, credential_id: str, owner_id: str) -> dict:
        """Decrypted secret. Only ever called server-side by a real tool
        executor at the moment it needs to make the external call — never
        by an API route, never stored back into run variables/trace."""
        record = self._get(credential_id, owner_id)
        try:
            raw = self._fernet.decrypt(record.encrypted_secret)
        except InvalidToken as exc:
            raise CredentialAccessError(
                f"credential '{credential_id}' cannot be decrypted with the current "
                "CREDENTIAL_ENCRYPTION_KEY (key rotated or process restarted without a durable key)"
            ) from exc
        return _deserialize(raw)

    def update_secret(self, credential_id: str, owner_id: str, secret: dict) -> CredentialSummary:
        """Replace the encrypted secret (e.g. after OAuth token refresh)."""
        record = self._get(credential_id, owner_id)
        validate_secret(record.provider, secret)
        encrypted = self._fernet.encrypt(_serialize(secret))
        with self._lock:
            record.encrypted_secret = encrypted
        return record.summary()

    def delete(self, credential_id: str, owner_id: str) -> None:
        with self._lock:
            record = self._credentials.get(credential_id)
            if record is None:
                raise CredentialNotFoundError(credential_id)
            if record.owner_id != owner_id:
                raise CredentialAccessError(credential_id)
            del self._credentials[credential_id]

    def validate_connection(self, credential_id: str, owner_id: str) -> ConnectionValidationResult:
        """Re-runs shape validation against the stored secret and records
        the outcome on the record (`status`, `last_validated_at`). This is
        the "Connection validation" leg of P0.1 — see `providers.py` for
        why this is a format check rather than a live network call in
        this sandbox, and how a real deployment upgrades it without
        changing this method's signature."""
        record = self._get(credential_id, owner_id)
        secret = self.get_secret(credential_id, owner_id)
        try:
            validate_secret(record.provider, secret)
        except CredentialValidationError as exc:
            self._update_status(credential_id, "invalid", str(exc))
            return ConnectionValidationResult(ok=False, error=str(exc))
        self._update_status(credential_id, "valid", None)
        return ConnectionValidationResult(ok=True)

    def validate_credential_ref(self, *, credential_id: str, tool_id: str, owner_id: str) -> None:
        """Raises if `credential_id` doesn't exist, isn't owned by
        `owner_id`, belongs to the wrong provider for `tool_id`, or was
        explicitly marked invalid by a prior validate_connection() call.

        That last check matters: existence + correct provider alone is
        NOT the same as "this integration is actually connected" — a
        credential whose secret was rejected the last time someone
        checked (status == "invalid") should not silently pass here just
        because a credential_ref happens to point at it. This backs "do
        not claim an integration is connected merely because a
        credentialRef exists": the compiler (via this check,
        compiler.py's _check_credential_refs) and every real tool
        executor (real_executors/tools.py) both treat a binding as usable
        only when its referenced credential is not known-bad — the
        default "unvalidated" status (true for most credentials, since
        this sandbox has no live egress to actually verify one against a
        real provider API — see credentials/providers.py) is still
        accepted, since requiring an explicit validate step for every
        credential before first use would make the credential store
        largely unusable here; only a credential someone already checked
        and got a concrete "invalid" result for is refused.
        """
        record = self._get(credential_id, owner_id)
        if record.status == "invalid":
            raise CredentialValidationError(
                f"credential '{credential_id}' was marked invalid the last time its connection was "
                f"checked ({record.last_validation_error or 'no details recorded'}) — fix or replace "
                f"it before using it in a connected tool binding."
            )
        expected_provider = provider_for_tool(tool_id)
        if expected_provider is None:
            # Tool has no registered provider mapping (e.g. a demo-only
            # tool) — nothing to check a credential against.
            return
        if record.provider != expected_provider:
            raise CredentialValidationError(
                f"credential '{credential_id}' is a '{record.provider}' credential, "
                f"but tool '{tool_id}' requires a '{expected_provider}' credential"
            )

    def _get(self, credential_id: str, owner_id: str) -> Credential:
        with self._lock:
            record = self._credentials.get(credential_id)
        if record is None:
            raise CredentialNotFoundError(credential_id)
        if record.owner_id != owner_id:
            # Deliberately the same error as "not found" from the caller's
            # point of view when wired into an HTTP 404 in app.py — a
            # credential owned by someone else should not be
            # distinguishable from a nonexistent one via enumeration.
            raise CredentialAccessError(credential_id)
        return record

    def _update_status(self, credential_id: str, status: str, error: str | None) -> None:
        from datetime import datetime, timezone

        with self._lock:
            record = self._credentials.get(credential_id)
            if record is None:
                return
            record.status = status  # type: ignore[assignment]
            record.last_validation_error = error
            record.last_validated_at = datetime.now(timezone.utc).isoformat()
            record.updated_at = record.last_validated_at


def _serialize(secret: dict) -> bytes:
    import json

    return json.dumps(secret).encode("utf-8")


def _deserialize(raw: bytes) -> dict:
    import json

    return json.loads(raw.decode("utf-8"))


# Process-wide singleton, same rationale as `backend.api.store.store`:
# fine for a single-worker dev server, swapped via a FastAPI dependency
# override for anything else.
credential_store = CredentialStore()
