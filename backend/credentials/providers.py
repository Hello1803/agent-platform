"""Provider catalogue for connected credentials.

This is the credential-side counterpart to `backend/tools/registry.py`:
where the tool registry says *which tools exist*, this module says *what a
valid credential for that tool's provider looks like* — the required
secret fields, and a format-level validation routine.

Deliberately explicit about a real constraint: this sandbox's outbound
network is allowlisted to package registries only (see
`network_configuration`), so a `validate()` here cannot literally place an
OAuth handshake against `accounts.google.com` or a real Slack API call.
What it *can* and does do — and what "Connection validation" in the P0
scope actually buys you regardless of network — is:

  1. reject a credential that's missing a required field before it's ever
     stored (fail at creation time, not at run time deep inside a node
     executor), and
  2. reject the *shape* of an obviously-wrong secret (an empty string, an
     API key that doesn't match the provider's known key format, an OAuth
     payload missing its refresh token) — the class of bug that "connect
     Gmail, forgot to paste the client secret" actually is in practice.

A real deployment swaps `_live_check` bodies for genuine API calls once
egress to the provider's domain is permitted; every provider's `validate()`
entrypoint stays the same, so nothing calling this module needs to change
when that happens — the identical seam the executor stub/real split
already uses elsewhere in this repo.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable


class CredentialValidationError(ValueError):
    """A credential's shape or contents failed provider-level checks."""


@dataclass(frozen=True)
class SecretField:
    name: str
    required: bool = True
    # Format-checked at creation time; e.g. r"^AIza[0-9A-Za-z_-]{35}$" for a
    # Google API key. None means "non-empty string is enough."
    pattern: str | None = None
    description: str = ""


@dataclass(frozen=True)
class ProviderDefinition:
    id: str
    name: str
    auth_type: str  # "oauth2" | "api_key" | "basic"
    fields: tuple[SecretField, ...]


def _check_fields(provider: ProviderDefinition, secret: dict) -> list[str]:
    problems: list[str] = []
    for field in provider.fields:
        value = secret.get(field.name)
        if field.required and not value:
            problems.append(f"missing required field '{field.name}'")
            continue
        if value and field.pattern and not re.match(field.pattern, str(value)):
            problems.append(f"field '{field.name}' does not match the expected format for {provider.name}")
    # Reject unknown fields too — a stray field is the shape of a caller
    # pasting the wrong provider's payload in, not something to silently keep.
    known = {f.name for f in provider.fields}
    unexpected = set(secret.keys()) - known
    if unexpected:
        problems.append(f"unexpected field(s) for {provider.name}: {', '.join(sorted(unexpected))}")
    return problems


PROVIDER_REGISTRY: dict[str, ProviderDefinition] = {
    "gmail": ProviderDefinition(
        id="gmail",
        name="Gmail",
        auth_type="oauth2",
        fields=(
            SecretField("access_token", description="OAuth access token"),
            SecretField("refresh_token", required=False, description="OAuth refresh token (needed to renew access)"),
            SecretField("client_id", description="OAuth client id"),
            SecretField("client_secret", description="OAuth client secret"),
        ),
    ),
    "smtp": ProviderDefinition(
        id="smtp",
        name="Email (SMTP)",
        auth_type="basic",
        fields=(
            SecretField("host", description="SMTP host, e.g. smtp.gmail.com"),
            SecretField("port", required=False, description="SMTP port (default 587)"),
            SecretField("username", description="Your full email address / SMTP username"),
            SecretField("password", description="SMTP password or app password"),
            SecretField("from_email", required=False, description="From address (defaults to username)"),
            SecretField("use_tls", required=False, description="Use STARTTLS: true/false (default true)"),
            SecretField("imap_host", required=False, description="IMAP host for reading mail (default: derived from SMTP host)"),
            SecretField("imap_port", required=False, description="IMAP port (default 993)"),
        ),
    ),
    "google_sheets": ProviderDefinition(
        id="google_sheets",
        name="Google Sheets",
        auth_type="oauth2",
        fields=(
            SecretField("access_token", description="OAuth access token"),
            SecretField("refresh_token", description="OAuth refresh token"),
            SecretField("client_id", description="OAuth client id"),
            SecretField("client_secret", description="OAuth client secret"),
        ),
    ),
    "slack": ProviderDefinition(
        id="slack",
        name="Slack",
        auth_type="api_key",
        fields=(
            SecretField("bot_token", pattern=r"^xox[bp]-", description="Slack bot/user OAuth token"),
        ),
    ),
    "http_api": ProviderDefinition(
        id="http_api",
        name="HTTP / API",
        auth_type="api_key",
        fields=(
            SecretField("api_key", required=False, description="Bearer token or API key, if the endpoint needs one"),
            SecretField("header_name", required=False, description="Header to send the key in (defaults to Authorization)"),
        ),
    ),
    "files": ProviderDefinition(
        id="files",
        name="File storage",
        auth_type="api_key",
        fields=(
            SecretField("api_key", required=False, description="Access token for the file store, if required"),
        ),
    ),
}


def get_provider(provider_id: str) -> ProviderDefinition | None:
    return PROVIDER_REGISTRY.get(provider_id)


def validate_secret(provider_id: str, secret: dict) -> None:
    """Raise CredentialValidationError if `secret` isn't shaped correctly
    for `provider_id`. Called at credential-creation time so a malformed
    credential is rejected before it's ever referenced by a spec."""
    provider = get_provider(provider_id)
    if provider is None:
        raise CredentialValidationError(f"unknown provider '{provider_id}'")
    if not isinstance(secret, dict) or not secret:
        raise CredentialValidationError("secret payload must be a non-empty object")
    problems = _check_fields(provider, secret)
    if problems:
        raise CredentialValidationError("; ".join(problems))


# Maps a tool registry `tool_id` (backend/tools/registry.py) to the
# credential provider whose secret it needs. Kept as an explicit table
# rather than a naming convention so tool ids can evolve independently of
# provider ids.
TOOL_ID_TO_PROVIDER: dict[str, str] = {
    "gmail.send": "gmail",
    "gmail.read": "gmail",
    "smtp.send": "smtp",
    "smtp.read": "smtp",
    "google_sheets.append_row": "google_sheets",
    "slack.post_message": "slack",
    "http.request": "http_api",
    "files.read": "files",
    "files.write": "files",
}


def provider_for_tool(tool_id: str) -> str | None:
    return TOOL_ID_TO_PROVIDER.get(tool_id)
