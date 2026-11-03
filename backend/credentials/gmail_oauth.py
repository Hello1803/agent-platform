"""Gmail OAuth2 authorization-code flow + access-token refresh.

Env (required for the live OAuth dance):
  GOOGLE_CLIENT_ID
  GOOGLE_CLIENT_SECRET
  GOOGLE_OAUTH_REDIRECT_URI  (default http://localhost:8000/credentials/oauth/gmail/callback)

Scopes used: gmail.send + gmail.readonly (covers Send Email and Read Email nodes).
"""
from __future__ import annotations

import os
import secrets
import time
from typing import Any
from urllib.parse import urlencode

import httpx

GMAIL_SCOPES = " ".join(
    [
        "https://www.googleapis.com/auth/gmail.send",
        "https://www.googleapis.com/auth/gmail.readonly",
        "https://www.googleapis.com/auth/userinfo.email",
        "openid",
    ]
)

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"

# Short-lived state → owner_id for CSRF protection (in-process; fine for single-server).
_pending_states: dict[str, dict[str, Any]] = {}


class GmailOAuthError(RuntimeError):
    pass


def _oauth_config_path() -> str:
    base = os.environ.get("AGENT_FILES_DIR", "agent_platform_files")
    return os.path.join(base, ".google_oauth.json")


def load_oauth_app_config() -> dict[str, str]:
    """Merge env + optional UI-saved config file (UI wins only when env is empty)."""
    import json
    from pathlib import Path as _P

    cfg: dict[str, str] = {
        "client_id": os.environ.get("GOOGLE_CLIENT_ID", "").strip(),
        "client_secret": os.environ.get("GOOGLE_CLIENT_SECRET", "").strip(),
        "redirect_uri": (
            os.environ.get("GOOGLE_OAUTH_REDIRECT_URI")
            or "http://localhost:8000/credentials/oauth/gmail/callback"
        ).strip().rstrip("/"),
    }
    path = _P(_oauth_config_path())
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                for key in ("client_id", "client_secret", "redirect_uri"):
                    val = str(data.get(key) or "").strip()
                    if val and not cfg.get(key):
                        cfg[key] = val.rstrip("/") if key == "redirect_uri" else val
                    elif val and key == "redirect_uri" and not os.environ.get("GOOGLE_OAUTH_REDIRECT_URI"):
                        cfg[key] = val.rstrip("/")
                    elif val and key in ("client_id", "client_secret") and not os.environ.get(
                        "GOOGLE_CLIENT_ID" if key == "client_id" else "GOOGLE_CLIENT_SECRET"
                    ):
                        cfg[key] = val
        except Exception:
            pass
    return cfg


def save_oauth_app_config(*, client_id: str, client_secret: str, redirect_uri: str | None = None) -> dict[str, str]:
    import json
    from pathlib import Path as _P

    path = _P(_oauth_config_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "client_id": client_id.strip(),
        "client_secret": client_secret.strip(),
        "redirect_uri": (
            redirect_uri
            or "http://localhost:8000/credentials/oauth/gmail/callback"
        ).strip().rstrip("/"),
    }
    if not data["client_id"] or not data["client_secret"]:
        raise GmailOAuthError("client_id and client_secret are required.")
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def _client_id() -> str:
    value = load_oauth_app_config().get("client_id", "")
    if not value:
        raise GmailOAuthError(
            "Google OAuth client_id is not set. Use Setup Gmail OAuth in the UI "
            "(or export GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET)."
        )
    return value


def _client_secret() -> str:
    value = load_oauth_app_config().get("client_secret", "")
    if not value:
        raise GmailOAuthError("Google OAuth client_secret is not set. Configure it in the UI or env.")
    return value


def redirect_uri() -> str:
    return load_oauth_app_config().get(
        "redirect_uri",
        "http://localhost:8000/credentials/oauth/gmail/callback",
    ).rstrip("/")


def build_authorize_url(*, owner_id: str = "default", credential_name: str = "Gmail") -> tuple[str, str]:
    """Returns (authorize_url, state)."""
    state = secrets.token_urlsafe(24)
    _pending_states[state] = {
        "owner_id": owner_id,
        "credential_name": credential_name,
        "created_at": time.time(),
    }
    params = {
        "client_id": _client_id(),
        "redirect_uri": redirect_uri(),
        "response_type": "code",
        "scope": GMAIL_SCOPES,
        "access_type": "offline",
        "prompt": "consent",  # ensure refresh_token is returned
        "state": state,
    }
    return f"{GOOGLE_AUTH_URL}?{urlencode(params)}", state


def pop_state(state: str) -> dict[str, Any]:
    meta = _pending_states.pop(state, None)
    if not meta:
        raise GmailOAuthError("Invalid or expired OAuth state. Start the Gmail connect flow again.")
    if time.time() - meta["created_at"] > 600:
        raise GmailOAuthError("OAuth state expired. Start the Gmail connect flow again.")
    return meta


def exchange_code_for_tokens(code: str) -> dict[str, Any]:
    data = {
        "code": code,
        "client_id": _client_id(),
        "client_secret": _client_secret(),
        "redirect_uri": redirect_uri(),
        "grant_type": "authorization_code",
    }
    with httpx.Client(timeout=20.0) as client:
        response = client.post(GOOGLE_TOKEN_URL, data=data)
    if response.status_code >= 400:
        raise GmailOAuthError(f"Token exchange failed ({response.status_code}): {response.text[:500]}")
    payload = response.json()
    if not payload.get("access_token"):
        raise GmailOAuthError(f"Token response missing access_token: {payload}")
    return {
        "access_token": payload["access_token"],
        "refresh_token": payload.get("refresh_token") or "",
        "client_id": _client_id(),
        "client_secret": _client_secret(),
        "expires_in": payload.get("expires_in"),
        "token_type": payload.get("token_type", "Bearer"),
        "scope": payload.get("scope", GMAIL_SCOPES),
    }


def refresh_access_token(secret: dict[str, Any]) -> dict[str, Any]:
    """Return an updated secret dict with a fresh access_token."""
    refresh = secret.get("refresh_token")
    client_id = secret.get("client_id") or os.environ.get("GOOGLE_CLIENT_ID", "")
    client_secret = secret.get("client_secret") or os.environ.get("GOOGLE_CLIENT_SECRET", "")
    if not refresh:
        raise GmailOAuthError("Gmail credential has no refresh_token — reconnect Gmail with prompt=consent.")
    if not client_id or not client_secret:
        raise GmailOAuthError("Gmail credential is missing client_id/client_secret for refresh.")
    data = {
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh,
        "grant_type": "refresh_token",
    }
    with httpx.Client(timeout=20.0) as client:
        response = client.post(GOOGLE_TOKEN_URL, data=data)
    if response.status_code >= 400:
        raise GmailOAuthError(f"Token refresh failed ({response.status_code}): {response.text[:500]}")
    payload = response.json()
    updated = dict(secret)
    updated["access_token"] = payload["access_token"]
    if payload.get("refresh_token"):
        updated["refresh_token"] = payload["refresh_token"]
    return updated
