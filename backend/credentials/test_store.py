"""
Regression tests for backend/credentials/store.py + providers.py — the
server-side credential store (P0.1: no secrets in AgentSpec/frontend,
connection validation).

Run:
    PYTHONPATH=. python backend/credentials/test_store.py
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

from backend.api.app import app
from backend.compiler.compiler import CompileError, compile_agent
from backend.credentials.providers import CredentialValidationError, validate_secret
from backend.credentials.store import (
    CredentialAccessError,
    CredentialNotFoundError,
    CredentialStore,
)
from backend.models.agent_spec import AgentSpecification

ROOT = Path(__file__).resolve().parents[2]


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_spec_json(name: str) -> dict:
    return json.loads((ROOT / "examples" / name).read_text())


def main() -> None:
    store = CredentialStore(fernet=Fernet(Fernet.generate_key()))

    print("=== create + encrypted-at-rest round trip ===")
    summary = store.create(
        provider="slack",
        name="Ops channel",
        secret={"bot_token": "xoxb-fake-token-123"},
        owner_id="alice",
    )
    check(summary.status == "unvalidated", "new credential should start unvalidated")
    check(not hasattr(summary, "secret"), "summary must never carry a 'secret' attribute")
    check("bot_token" not in summary.model_dump(), "summary dump must not leak the secret field")
    print(f"  OK: created {summary.id}, summary carries no secret material")

    fetched = store.get_secret(summary.id, "alice")
    check(fetched["bot_token"] == "xoxb-fake-token-123", "owner should be able to decrypt their own secret")
    print("  OK: owner can decrypt their own secret via get_secret()")

    print("\n=== owner isolation ===")
    try:
        store.get_secret(summary.id, "mallory")
        raise AssertionError("a different owner should not be able to read this credential")
    except CredentialAccessError:
        print("  OK: a different owner cannot read/decrypt someone else's credential")

    try:
        store.get_summary(summary.id, "mallory")
        raise AssertionError("a different owner should not see even the summary")
    except CredentialAccessError:
        print("  OK: a different owner cannot even see the summary")

    print("\n=== shape validation at creation time ===")
    try:
        store.create(provider="slack", name="bad", secret={"bot_token": "not-a-slack-token"}, owner_id="alice")
        raise AssertionError("malformed slack token should be rejected")
    except CredentialValidationError:
        print("  OK: malformed provider-specific token format rejected before storage")

    try:
        store.create(provider="gmail", name="incomplete", secret={"access_token": "x"}, owner_id="alice")
        raise AssertionError("gmail credential missing required OAuth fields should be rejected")
    except CredentialValidationError:
        print("  OK: incomplete OAuth payload rejected before storage")

    print("\n=== connection validation ===")
    result = store.validate_connection(summary.id, "alice")
    check(result.ok, "a well-formed slack credential should validate OK")
    refreshed = store.get_summary(summary.id, "alice")
    check(refreshed.status == "valid", "status should flip to 'valid' after validate_connection")
    check(refreshed.last_validated_at is not None, "last_validated_at should be set")
    print("  OK: validate_connection() flips status to 'valid' and stamps last_validated_at")

    print("\n=== deletion ===")
    store.delete(summary.id, "alice")
    try:
        store.get_summary(summary.id, "alice")
        raise AssertionError("credential should be gone after delete")
    except CredentialNotFoundError:
        print("  OK: deleted credential is gone")

    print("\n=== compiler integration: credential_ref must exist and match provider ===")
    gmail_cred = store.create(
        provider="gmail",
        name="Support inbox",
        secret={
            "access_token": "a",
            "refresh_token": "r",
            "client_id": "c",
            "client_secret": "s",
        },
        owner_id="bob",
    )

    invoice_spec = load_spec_json("invoice-agent.json")
    invoice_spec["metadata"]["owner"] = "bob"
    # Point the send_email tool binding at a real, owned, correct-provider credential.
    for tool in invoice_spec["tools"]:
        if tool["toolId"] == "gmail.send":
            tool["credentialRef"] = gmail_cred.id
            tool["mode"] = "connected"
    spec = AgentSpecification.model_validate(invoice_spec)
    compile_agent(spec, credential_store=store)
    print("  OK: spec compiles when credential_ref points at a real, owned, correctly-provider'd credential")

    bad_spec_dict = copy.deepcopy(invoice_spec)
    for tool in bad_spec_dict["tools"]:
        if tool["toolId"] == "gmail.send":
            tool["credentialRef"] = "cred_does_not_exist"
    bad_spec = AgentSpecification.model_validate(bad_spec_dict)
    try:
        compile_agent(bad_spec, credential_store=store)
        raise AssertionError("compiling with a nonexistent credential_ref should fail")
    except CompileError as exc:
        check("does not exist" in str(exc), f"unexpected error message: {exc}")
        print("  OK: compile_agent rejects a credential_ref that doesn't exist")

    # A credential that exists but belongs to the wrong provider (e.g. a
    # Slack token bound to a tool that needs Gmail).
    wrong_provider_spec = copy.deepcopy(invoice_spec)
    slack_cred = store.create(provider="slack", name="wrong provider", secret={"bot_token": "xoxb-abc"}, owner_id="bob")
    for tool in wrong_provider_spec["tools"]:
        if tool["toolId"] == "gmail.send":
            tool["credentialRef"] = slack_cred.id
    wp_spec = AgentSpecification.model_validate(wrong_provider_spec)
    try:
        compile_agent(wp_spec, credential_store=store)
        raise AssertionError("compiling with a wrong-provider credential should fail")
    except CompileError as exc:
        check("requires a 'gmail' credential" in str(exc), f"unexpected error message: {exc}")
        print("  OK: compile_agent rejects a credential of the wrong provider for the tool")

    print("\n=== compiler integration: a credential existing + right provider is NOT enough if it's known-invalid ===")
    # This is the concrete fix for "do not claim an integration is
    # connected merely because a credentialRef exists": existence and a
    # matching provider used to be the ONLY things checked -- a
    # credential explicitly marked invalid by a prior validate_connection()
    # call (e.g. its secret was found malformed) still passed. Now it
    # doesn't.
    flaky_cred = store.create(
        provider="gmail",
        name="Revoked token",
        secret={"access_token": "a", "refresh_token": "r", "client_id": "c", "client_secret": "s"},
        owner_id="bob",
    )
    store._update_status(flaky_cred.id, "invalid", "refresh token revoked upstream")
    invalid_cred_spec_dict = copy.deepcopy(invoice_spec)
    for tool in invalid_cred_spec_dict["tools"]:
        if tool["toolId"] == "gmail.send":
            tool["credentialRef"] = flaky_cred.id
    invalid_cred_spec = AgentSpecification.model_validate(invalid_cred_spec_dict)
    try:
        compile_agent(invalid_cred_spec, credential_store=store)
        raise AssertionError("compiling with a credential explicitly marked invalid should fail")
    except CompileError as exc:
        check("marked invalid" in str(exc), f"unexpected error message: {exc}")
        print(f"  OK: compile_agent rejects a credential that exists, matches the provider, but is marked invalid: {exc}")

    print("\n=== real HTTP API: secrets never appear in a response ===")
    client = TestClient(app)
    resp = client.post(
        "/credentials",
        json={"provider": "slack", "name": "Alerts", "secret": {"bot_token": "xoxb-http-test"}},
        headers={"X-User-Id": "carol"},
    )
    check(resp.status_code == 201, f"expected 201, got {resp.status_code}: {resp.text}")
    body = resp.json()
    check("secret" not in body and "bot_token" not in json.dumps(body), "API response must not leak the secret")
    cred_id = body["id"]
    print(f"  OK: POST /credentials created {cred_id}, response body carries no secret material")

    resp = client.get("/credentials", headers={"X-User-Id": "carol"})
    check(resp.status_code == 200 and any(c["id"] == cred_id for c in resp.json()), "carol should see her own credential")
    print("  OK: GET /credentials lists it back for the owner")

    resp = client.get("/credentials", headers={"X-User-Id": "dave"})
    check(resp.status_code == 200 and all(c["id"] != cred_id for c in resp.json()), "dave should not see carol's credential")
    print("  OK: GET /credentials does not leak another owner's credentials")

    resp = client.get(f"/credentials/{cred_id}", headers={"X-User-Id": "dave"})
    check(resp.status_code == 404, "a different owner fetching by id should get 404, not 403 (no enumeration signal)")
    print("  OK: cross-owner GET by id returns 404, not a distinguishable 403")

    resp = client.post(f"/credentials/{cred_id}/validate", headers={"X-User-Id": "carol"})
    check(resp.status_code == 200 and resp.json()["status"] == "valid", "validate should succeed for a well-formed secret")
    print("  OK: POST /credentials/{id}/validate works end-to-end over real HTTP")

    print("\nAll credential store checks passed.")


if __name__ == "__main__":
    main()
