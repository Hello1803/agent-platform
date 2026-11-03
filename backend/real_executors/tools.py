"""
Mode-aware real executors for tool-calling node types: search_web,
send_email, read_email, read_file, write_file, api_call.

**The bug this closes**: `backend/compiler/executors.py`'s stub functions
return identical canned "success" data — `{"email_sent": True}`,
`{"file_contents": "[stub file contents]"}` — no matter what. A node whose
ToolBinding is `mode: "connected"` with a real, validated
`credential_ref` (see backend/credentials/, P0.1) still "ran" successfully
and reported `email_sent: True` without ever making a real call. That is
the "silent stub fallback" the roadmap's Runtime Executors item names —
not a hypothetical, this repo's own `backend/compiler/executors.py` had
it, unconditionally, in every stub function that touches an external
service.

Each executor below:

  1. Resolves which `ToolBinding` on the node actually grants the
     capability it needs — same lookup shape as compiler.py's
     `_check_tool_permissions` / executors.py's `_require_tool`, just
     keeping the binding object (for its `.mode` and `.credential_ref`)
     instead of only checking permission presence.
  2. `mode == "demo"`: delegates to the exact stub function in
     `backend.compiler.executors`. Demo is a deliberate, honest choice not
     to touch a live integration — it behaves identically whether or not
     a real credential happens to exist, so flipping a tool binding's mode
     back and forth between demo/connected never has a surprising effect
     on a demo run.
  3. `mode == "connected"`: performs the real external call, using the
     secret resolved from `backend.credentials.store` at the moment of the
     call (not cached, not resolved at compile time — see
     credentials/store.py's docstring on why). A failure here (network
     unreachable, bad/expired credential, non-2xx response) RAISES. It is
     never caught and converted into a fake success — the API layer's
     existing `RuntimeError -> RunResponse(status="failed")` handling
     (backend/api/app.py's /run and /run/stream routes) is what already
     surfaces this to the caller; no new plumbing needed.

**Network reality check** (see this repo's `network_configuration`): this
sandbox's outbound egress is allowlisted to package registries only, not
`gmail.googleapis.com` / `slack.com` / `sheets.googleapis.com`. The Gmail/
Slack/Sheets request-building below is genuinely correct — proven with an
injectable `httpx` transport in `test_tools.py`, the same pattern the NL
compiler's tests use a fake Anthropic client for — but can't be proven
against the *real* provider APIs from here. `api_call` (fully generic
HTTP) and `read_file`/`write_file` need no disallowed domain and ARE
proven end-to-end against real, allowed egress in that same test file.
"""
from __future__ import annotations

import json
import mimetypes
import os
from pathlib import Path
from typing import Any, Callable

import httpx

from backend.compiler.executors import (
    NodeExecutor,
    ToolPermissionError,
    exec_generic_tool,
    exec_read_email,
    exec_read_file,
    exec_send_email,
    exec_write_file,
)
from backend.credentials.store import CredentialStore, credential_store as _default_credential_store
from backend.models.agent_spec import AgentNode, AgentSpecification, ToolBinding


class ConnectedExecutionError(RuntimeError):
    """Raised when a mode='connected' tool call genuinely fails — a real
    attempt was made and it didn't work, as opposed to a stub silently
    reporting success. Deliberately a RuntimeError subclass so it's
    caught by the exact same failure path the API layer already has
    (see backend/api/app.py's /run route)."""


def _resolve_binding(node: AgentNode, spec: AgentSpecification, capability: str) -> ToolBinding:
    """The binding-returning counterpart to executors.py's `_require_tool`
    — same permission re-check (compiler already verified this at compile
    time; this is defense-in-depth for credentials/permissions that
    changed between compile and run), but returns the actual binding so
    its `.mode` and `.credential_ref` are available to the caller."""
    if not node.tool_refs:
        raise ToolPermissionError(f"Node '{node.id}' has no tool binding but tried to use one.")
    bindings_by_id = {t.id: t for t in spec.tools}
    for ref in node.tool_refs:
        binding = bindings_by_id.get(ref)
        if binding and capability in binding.permissions:
            return binding
    raise ToolPermissionError(
        f"Node '{node.id}' is not permitted capability '{capability}' (tool_refs={node.tool_refs})."
    )


def _try_resolve_binding(node: AgentNode, spec: AgentSpecification, capability: str) -> ToolBinding | None:
    """Like _resolve_binding but returns None when no binding is configured."""
    if not node.tool_refs:
        return None
    try:
        return _resolve_binding(node, spec, capability)
    except Exception:
        return None



def _get_secret(store: CredentialStore, spec: AgentSpecification, binding: ToolBinding) -> dict:
    if not binding.credential_ref:
        raise ConnectedExecutionError(
            f"Tool binding '{binding.id}' is mode='connected' but has no credentialRef configured."
        )
    owner_id = spec.metadata.owner or "default"
    return store.get_secret(binding.credential_ref, owner_id)


def _http(method: str, url: str, *, transport: httpx.BaseTransport | None = None, **kwargs) -> httpx.Response:
    """Every real HTTP call in this module goes through here so tests can
    inject a fake `transport` (httpx.MockTransport) instead of needing
    live egress to a disallowed domain — see test_tools.py."""
    with httpx.Client(transport=transport, timeout=15.0) as client:
        response = client.request(method, url, **kwargs)
    if response.status_code >= 400:
        raise ConnectedExecutionError(
            f"{method} {url} returned {response.status_code}: {response.text[:500]}"
        )
    return response


# ---- send_email (Gmail) --------------------------------------------------

def _looks_like_email(value: object) -> bool:
    if not isinstance(value, str):
        return False
    s = value.strip()
    return "@" in s and "." in s.split("@")[-1] and " " not in s and len(s) < 320


def _first_email_in_text(text: str) -> str | None:
    import re
    if not text:
        return None
    # Prefer explicit To: lines from model drafts
    for pattern in (
        r"(?i)\bto\s*[:\-]\s*<?([A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})>?",
        r"(?i)\brecipient\s*[:\-]\s*<?([A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})>?",
        r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}",
    ):
        match = re.search(pattern, text)
        if match:
            return match.group(1) if match.lastindex else match.group(0)
    return None


def _email_from_row(row: dict, email_field: str | None) -> str | None:
    keys = []
    if email_field:
        keys.append(email_field)
    keys.extend(["Email", "email", "EMAIL", "mail", "E-mail", "e-mail"])
    for key in keys:
        if row.get(key) and _looks_like_email(str(row[key])):
            return str(row[key]).strip()
    for value in row.values():
        if _looks_like_email(value):
            return str(value).strip()
    return None


def _resolve_email_recipient(node: AgentNode, variables: dict) -> str:
    """Recipient resolution: config.to → variables → loop item → table → AI text."""
    to_addr = node.config.get("to") or variables.get("to") or variables.get("recipient_email")
    if to_addr and _looks_like_email(str(to_addr)):
        return str(to_addr).strip()
    if to_addr and "@" in str(to_addr):
        return str(to_addr).strip()

    email_field = node.config.get("email_field") or "Email"
    item = variables.get("current_item") or variables.get("item")
    if isinstance(item, dict):
        found = _email_from_row(item, email_field)
        if found:
            return found

    # First row of a spreadsheet read
    for key in ("table_preview", "rows"):
        rows = variables.get(key)
        if isinstance(rows, list) and rows:
            first = rows[0]
            if isinstance(first, dict):
                found = _email_from_row(first, email_field)
                if found:
                    return found

    # AI / agent prose often contains "To: someone@x.com" or a bare address.
    # Also scan last_ai_prompt — users often put the address only in the Ask AI prompt.
    for key in ("draft_email", "ai_response", "summary", "email_body", "last_ai_prompt", "prompt"):
        raw = variables.get(key)
        if isinstance(raw, str) and raw.strip():
            found = _first_email_in_text(raw)
            if found:
                return found

    # Last resort: any string variable that contains an email
    for key, raw in list(variables.items()):
        if key in ("draft_email", "ai_response", "summary", "email_body", "last_ai_prompt", "prompt"):
            continue
        if isinstance(raw, str) and "@" in raw:
            found = _first_email_in_text(raw)
            if found:
                return found

    raise ConnectedExecutionError(
        f"Node '{node.id}' is mode='connected' send_email but has no recipient — "
        "set the To field on Send Email (e.g. you@gmail.com), or put an Email column "
        "in the file / Loop current_item, or include an address in the AI draft."
    )


def _template_context(variables: dict) -> dict:
    """Flatten common sources so {{Name}} works without a Loop."""
    ctx: dict = dict(variables)

    def _ingest_row(row: dict, *, prefix: str | None = None) -> None:
        for k, v in row.items():
            key = str(k).strip()
            ctx[key] = v
            ctx[key.lower()] = v
            ctx[key.replace(" ", "_")] = v
            ctx[key.replace(" ", "_").lower()] = v
            if prefix:
                ctx[f"{prefix}.{key}"] = v
                ctx[f"{prefix}.{key.lower()}"] = v

    item = variables.get("current_item") or variables.get("item")
    if isinstance(item, dict):
        _ingest_row(item, prefix="current_item")

    for key in ("table_preview", "rows"):
        rows = variables.get(key)
        if isinstance(rows, list) and rows and isinstance(rows[0], dict):
            _ingest_row(rows[0])
            break

    # Columns listed without values still help debugging; skip.
    # item_Name style keys written by the loop executor
    for k, v in list(variables.items()):
        if isinstance(k, str) and k.startswith("item_") and len(k) > 5:
            field = k[5:]
            ctx.setdefault(field, v)
            ctx.setdefault(field.lower(), v)

    # Parse JSON blob from Ask AI (object, array, or filtered_students list)
    import json as _json
    import re as _re

    def _try_parse_json_blob(raw: str):
        raw = raw.strip()
        if "```" in raw:
            parts = raw.split("```")
            for part in parts[1::2]:
                chunk = part.strip()
                if chunk.lower().startswith("json"):
                    chunk = chunk[4:].lstrip()
                try:
                    return _json.loads(chunk)
                except Exception:
                    pass
        for match in _re.finditer(r"[\[{]", raw):
            try:
                return _json.loads(raw[match.start():])
            except Exception:
                continue
        return None

    for key in ("ai_response", "draft_email", "summary"):
        raw = variables.get(key)
        if not isinstance(raw, str) or not raw.strip():
            continue
        parsed = _try_parse_json_blob(raw)
        if parsed is None:
            continue
        if isinstance(parsed, dict):
            students = parsed.get("filtered_students") or parsed.get("students")
            if isinstance(students, list) and students and isinstance(students[0], dict):
                _ingest_row(students[0])
            else:
                _ingest_row(parsed)
        elif isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
            _ingest_row(parsed[0])

    return ctx


def _lookup_path(ctx: dict, path: str):
    parts = [p for p in path.strip().split(".") if p]
    if not parts:
        return None
    # Exact then case-insensitive for the first segment
    if parts[0] in ctx:
        cur: object = ctx[parts[0]]
    else:
        lower_map = {str(k).lower(): v for k, v in ctx.items()}
        key = parts[0].lower()
        if key not in lower_map:
            return None
        cur = lower_map[key]
    for part in parts[1:]:
        if isinstance(cur, dict):
            if part in cur:
                cur = cur[part]
            else:
                lower_map = {str(k).lower(): v for k, v in cur.items()}
                if part.lower() not in lower_map:
                    return None
                cur = lower_map[part.lower()]
        else:
            return None
    return cur


def _render_body_template(template: str, variables: dict) -> str:
    """Substitute {{Name}}, {{current_item.Grade}}, {{Marks}}, etc.

    Values are taken from the shared variable bag, the Loop current_item,
    or the first spreadsheet row — so templates work even without a Loop.
    """
    import re

    ctx = _template_context(variables)

    def repl(match: re.Match) -> str:
        path = match.group(1).strip()
        value = _lookup_path(ctx, path)
        if value is None:
            return match.group(0)  # leave unknown placeholders visible
        return "" if value is None else str(value)

    return re.sub(r"\{\{\s*([^}]+?)\s*\}\}", repl, template)



def _resolve_email_body(node: AgentNode, variables: dict) -> str:
    """Choose the email body from AI output by default.

    Priority:
      1. body_variable if set and present
      2. non-empty node.config.body (optional user template, still {{}}-rendered)
      3. draft_email (AI Agent) / ai_response (Ask AI) / summary
    Empty config.body is ignored so clearing the template uses the model text.
    """
    body_var = node.config.get("body_variable")
    if body_var and variables.get(body_var) not in (None, ""):
        body = variables.get(body_var)
    else:
        configured = node.config.get("body")
        if isinstance(configured, str) and configured.strip():
            body = configured
        else:
            body = (
                variables.get("email_body")
                or variables.get("draft_email")
                or variables.get("ai_response")
                or variables.get("summary")
                or ""
            )
    return _render_body_template(str(body or ""), variables)


def _connected_send_email(
    node: AgentNode,
    spec: AgentSpecification,
    variables: dict,
    secret: dict,
    transport=None,
    *,
    credential_store: CredentialStore | None = None,
    credential_id: str | None = None,
    owner_id: str = "default",
) -> dict:
    import base64
    from email.mime.text import MIMEText

    to_addr = _resolve_email_recipient(node, variables)

    body = _resolve_email_body(node, variables)
    if not body.strip():
        raise ConnectedExecutionError(
            f"Node '{node.id}' send_email has no body — set a Body template, or let Ask AI / AI Agent produce draft_email / ai_response."
        )
    subject = _render_body_template(
        str(node.config.get("subject") or "Message from agent"),
        variables,
    )

    mime = MIMEText(body)
    mime["to"] = to_addr
    mime["subject"] = subject
    raw = base64.urlsafe_b64encode(mime.as_bytes()).decode("ascii")

    access_token = secret.get("access_token")
    if not access_token:
        raise ConnectedExecutionError("Gmail credential is missing 'access_token'.")

    def _do_send(token: str):
        return _http(
            "POST",
            "https://gmail.googleapis.com/gmail/v1/users/me/messages/send",
            headers={"Authorization": f"Bearer {token}"},
            json={"raw": raw},
            transport=transport,
        )

    try:
        response = _do_send(access_token)
    except ConnectedExecutionError as exc:
        # Likely expired access token — refresh once and retry.
        if "401" not in str(exc) and "403" not in str(exc):
            raise
        try:
            from backend.credentials.gmail_oauth import refresh_access_token
            updated = refresh_access_token(secret)
        except Exception as refresh_exc:
            raise ConnectedExecutionError(
                f"Gmail send failed and token refresh failed: {refresh_exc}"
            ) from refresh_exc
        if credential_store and credential_id:
            try:
                credential_store.update_secret(credential_id, owner_id, {
                    "access_token": updated["access_token"],
                    "refresh_token": updated.get("refresh_token") or secret.get("refresh_token") or "",
                    "client_id": updated.get("client_id") or secret.get("client_id") or "",
                    "client_secret": updated.get("client_secret") or secret.get("client_secret") or "",
                })
            except Exception:
                pass
        response = _do_send(updated["access_token"])

    body_json = response.json()
    return {
        "email_sent": True,
        "email_message_id": body_json.get("id"),
        "email_to": to_addr,
        "email_subject": subject,
    }


def _connected_send_email_smtp(node: AgentNode, spec: AgentSpecification, variables: dict, secret: dict) -> dict:
    """Send via plain SMTP (Gmail app password, Outlook, corporate SMTP, etc.)."""
    import smtplib
    from email.mime.text import MIMEText

    to_addr = _resolve_email_recipient(node, variables)
    body = _resolve_email_body(node, variables)
    if not body.strip():
        raise ConnectedExecutionError(
            f"Node '{node.id}' send_email has no body — set a Body template, or let Ask AI / AI Agent produce draft_email / ai_response."
        )
    subject = _render_body_template(
        str(node.config.get("subject") or "Message from agent"),
        variables,
    )

    host = secret.get("host") or "smtp.gmail.com"
    port = int(secret.get("port") or 587)
    username = secret.get("username") or ""
    password = secret.get("password") or ""
    from_email = secret.get("from_email") or username
    use_tls = str(secret.get("use_tls", "true")).lower() not in ("0", "false", "no")

    if not username or not password:
        raise ConnectedExecutionError("SMTP credential needs username and password (use an app password for Gmail).")

    mime = MIMEText(body)
    mime["From"] = from_email
    mime["To"] = to_addr
    mime["Subject"] = subject

    try:
        if port == 465:
            with smtplib.SMTP_SSL(host, port, timeout=30) as smtp:
                smtp.login(username, password)
                smtp.sendmail(from_email, [to_addr], mime.as_string())
        else:
            with smtplib.SMTP(host, port, timeout=30) as smtp:
                if use_tls:
                    smtp.starttls()
                smtp.login(username, password)
                smtp.sendmail(from_email, [to_addr], mime.as_string())
    except Exception as exc:
        raise ConnectedExecutionError(f"SMTP send failed: {exc}") from exc

    return {"email_sent": True, "email_to": to_addr, "email_subject": subject, "email_via": "smtp"}


def build_send_email_executor(*, credential_store: CredentialStore | None = None, transport=None) -> NodeExecutor:
    store = credential_store or _default_credential_store

    def execute(node: AgentNode, spec: AgentSpecification, variables: dict[str, Any]) -> dict:
        binding = _resolve_binding(node, spec, "send")
        if binding.mode == "demo":
            return exec_send_email(node, spec, variables)
        secret = _get_secret(store, spec, binding)
        owner_id = spec.metadata.owner or "default"
        # Detect provider from the stored credential (Gmail OAuth vs SMTP).
        try:
            summary = store.get_summary(binding.credential_ref, owner_id)
            provider = summary.provider
        except Exception:
            provider = "gmail"
        if provider == "smtp":
            return _connected_send_email_smtp(node, spec, variables, secret)
        return _connected_send_email(
            node,
            spec,
            variables,
            secret,
            transport=transport,
            credential_store=store,
            credential_id=binding.credential_ref,
            owner_id=owner_id,
        )

    return execute


# ---- read_email (Gmail) ---------------------------------------------------

def _connected_read_email(node: AgentNode, spec: AgentSpecification, variables: dict, secret: dict, transport=None) -> dict:
    access_token = secret.get("access_token")
    if not access_token:
        raise ConnectedExecutionError("Gmail credential is missing 'access_token'.")
    query = node.config.get("query", "is:unread")

    list_response = _http(
        "GET",
        "https://gmail.googleapis.com/gmail/v1/users/me/messages",
        headers={"Authorization": f"Bearer {access_token}"},
        params={"q": query, "maxResults": 1},
        transport=transport,
    )
    messages = list_response.json().get("messages", [])
    if not messages:
        return {"email": None}

    message_id = messages[0]["id"]
    detail_response = _http(
        "GET",
        f"https://gmail.googleapis.com/gmail/v1/users/me/messages/{message_id}",
        headers={"Authorization": f"Bearer {access_token}"},
        params={"format": "metadata", "metadataHeaders": ["Subject", "From"]},
        transport=transport,
    )
    detail = detail_response.json()
    headers = {h["name"]: h["value"] for h in detail.get("payload", {}).get("headers", [])}
    return {
        "email": {
            "subject": headers.get("Subject", ""),
            "from": headers.get("From", ""),
            "snippet": detail.get("snippet", ""),
        }
    }


def _default_imap_host(smtp_host: str) -> str:
    host = (smtp_host or "").strip().lower()
    mapping = {
        "smtp.gmail.com": "imap.gmail.com",
        "smtp.office365.com": "outlook.office365.com",
        "smtp-mail.outlook.com": "outlook.office365.com",
        "smtp.mail.yahoo.com": "imap.mail.yahoo.com",
    }
    if host in mapping:
        return mapping[host]
    if host.startswith("smtp."):
        return "imap." + host[len("smtp."):]
    return host or "imap.gmail.com"


def _connected_read_email_imap(node: AgentNode, spec: AgentSpecification, variables: dict, secret: dict) -> dict:
    """Read latest matching message via IMAP (same app password as SMTP)."""
    import email as email_lib
    import imaplib
    from email.header import decode_header

    username = secret.get("username") or ""
    password = secret.get("password") or ""
    if not username or not password:
        raise ConnectedExecutionError("IMAP/SMTP credential needs username and password (app password for Gmail).")

    imap_host = secret.get("imap_host") or _default_imap_host(secret.get("host") or "smtp.gmail.com")
    imap_port = int(secret.get("imap_port") or 993)
    folder = node.config.get("folder") or "INBOX"
    limit = int(node.config.get("limit") or 1)
    query = (node.config.get("query") or "UNSEEN").strip()

    def _decode(value) -> str:
        if value is None:
            return ""
        parts = decode_header(value)
        out = []
        for part, enc in parts:
            if isinstance(part, bytes):
                out.append(part.decode(enc or "utf-8", errors="replace"))
            else:
                out.append(part)
        return "".join(out)

    try:
        client = imaplib.IMAP4_SSL(imap_host, imap_port)
        client.login(username, password)
        client.select(folder)
        # Simple IMAP search: UNSEEN / ALL / or raw criteria from config
        criteria = query if query.upper() in ("UNSEEN", "ALL", "SEEN", "RECENT") else "ALL"
        status, data = client.search(None, criteria)
        if status != "OK":
            raise ConnectedExecutionError(f"IMAP search failed: {status}")
        ids = data[0].split() if data and data[0] else []
        if not ids:
            client.logout()
            return {"email": None, "emails": []}

        # BODY.PEEK[] reads without setting \Seen (RFC822 fetch marks read on Gmail).
        mark_as_read = bool(node.config.get("mark_as_read", False))
        emails = []
        for msg_id in ids[-limit:]:
            status, msg_data = client.fetch(msg_id, "(BODY.PEEK[])")
            if status != "OK" or not msg_data or not msg_data[0]:
                continue
            raw = msg_data[0][1]
            msg = email_lib.message_from_bytes(raw)
            body_text = ""
            if msg.is_multipart():
                for part in msg.walk():
                    ctype = part.get_content_type()
                    if ctype == "text/plain" and "attachment" not in str(part.get("Content-Disposition", "")):
                        payload = part.get_payload(decode=True) or b""
                        body_text = payload.decode(part.get_content_charset() or "utf-8", errors="replace")
                        break
            else:
                payload = msg.get_payload(decode=True) or b""
                body_text = payload.decode(msg.get_content_charset() or "utf-8", errors="replace")
            emails.append(
                {
                    "subject": _decode(msg.get("Subject")),
                    "from": _decode(msg.get("From")),
                    "to": _decode(msg.get("To")),
                    "date": msg.get("Date") or "",
                    "snippet": (body_text or "")[:500],
                    "body": body_text,
                }
            )
            if mark_as_read:
                try:
                    client.store(msg_id, "+FLAGS", "\\Seen")
                except Exception:
                    pass
        client.logout()
    except ConnectedExecutionError:
        raise
    except Exception as exc:
        raise ConnectedExecutionError(f"IMAP read failed: {exc}") from exc

    latest = emails[-1] if emails else None
    return {"email": latest, "emails": emails}


def build_read_email_executor(*, credential_store: CredentialStore | None = None, transport=None) -> NodeExecutor:
    store = credential_store or _default_credential_store

    def execute(node: AgentNode, spec: AgentSpecification, variables: dict[str, Any]) -> dict:
        binding = _resolve_binding(node, spec, "read")
        if binding.mode == "demo":
            return exec_read_email(node, spec, variables)
        secret = _get_secret(store, spec, binding)
        owner_id = spec.metadata.owner or "default"
        try:
            provider = store.get_summary(binding.credential_ref, owner_id).provider
        except Exception:
            provider = "gmail"
        if provider == "smtp":
            return _connected_read_email_imap(node, spec, variables, secret)
        return _connected_read_email(node, spec, variables, secret, transport=transport)

    return execute


# ---- database / google_sheets.append_row ----------------------------------

def _connected_sheets_append(node: AgentNode, spec: AgentSpecification, variables: dict, secret: dict, transport=None) -> dict:
    access_token = secret.get("access_token")
    if not access_token:
        raise ConnectedExecutionError("Google Sheets credential is missing 'access_token'.")
    spreadsheet_id = node.config.get("spreadsheet_id")
    sheet_range = node.config.get("range", "Sheet1!A1")
    if not spreadsheet_id:
        raise ConnectedExecutionError(
            f"Node '{node.id}' is mode='connected' database (Google Sheets) but has no "
            "node.config.spreadsheet_id configured."
        )
    fields = node.config.get("fields") or list(variables.keys())
    row = [variables.get(f) for f in fields]

    response = _http(
        "POST",
        f"https://sheets.googleapis.com/v4/spreadsheets/{spreadsheet_id}/values/{sheet_range}:append",
        headers={"Authorization": f"Bearer {access_token}"},
        params={"valueInputOption": "RAW"},
        json={"values": [row]},
        transport=transport,
    )
    return {"saved": True, "sheets_response": response.json().get("updates", {})}


def build_database_connected_executor(*, credential_store: CredentialStore | None = None, transport=None, demo_executor: NodeExecutor | None = None) -> NodeExecutor:
    """Wraps a demo `database` executor (e.g. the real SQLite one from
    real_executors/database.py, or the plain stub) with mode-aware
    dispatch — a mode='connected' database node backed by
    google_sheets.append_row now genuinely calls Sheets instead of
    silently writing to SQLite (or worse, just returning `{"saved": True}`
    with nothing behind it)."""
    store = credential_store or _default_credential_store
    fallback_demo: NodeExecutor = demo_executor or (lambda node, spec, variables: {"saved": True})

    def execute(node: AgentNode, spec: AgentSpecification, variables: dict[str, Any]) -> dict:
        binding = _resolve_binding(node, spec, "write")
        if binding.mode == "demo":
            return fallback_demo(node, spec, variables)
        secret = _get_secret(store, spec, binding)
        return _connected_sheets_append(node, spec, variables, secret, transport=transport)

    return execute


# ---- api_call (generic HTTP / http_api provider) ---------------------------

def _render_http_string(template: str, variables: dict) -> str:
    """Replace {{var}} and $var in URL/header/body strings."""
    import re

    def repl_brace(match):
        key = match.group(1).strip()
        if key in variables:
            return str(variables[key])
        # dotted path
        parts = key.split(".")
        cur = variables
        for p in parts:
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                return match.group(0)
        return "" if cur is None else str(cur)

    out = re.sub(r"\{\{\s*([^}]+?)\s*\}\}", repl_brace, str(template))
    out = re.sub(r"\$([A-Za-z_][A-Za-z0-9_]*)", lambda m: str(variables[m.group(1)]) if m.group(1) in variables else m.group(0), out)
    return out


def _resolve_http_body(body, variables: dict):
    if body is None or body == "":
        return None
    if isinstance(body, str):
        rendered = _render_http_string(body, variables)
        try:
            import json as _json
            return _json.loads(rendered)
        except Exception:
            return rendered
    if isinstance(body, dict):
        resolved = {}
        for k, v in body.items():
            if isinstance(v, str) and v.startswith("$") and v[1:] in variables:
                resolved[k] = variables[v[1:]]
            elif isinstance(v, str) and "{{" in v:
                resolved[k] = _render_http_string(v, variables)
            else:
                resolved[k] = v
        return resolved
    return body


def _connected_api_call(node: AgentNode, spec: AgentSpecification, variables: dict, secret: dict | None, transport=None) -> dict:
    url = node.config.get("url")
    if not url:
        raise ConnectedExecutionError(f"Node '{node.id}' HTTP/api_call has no URL configured.")
    url = _render_http_string(str(url), variables)
    method = str(node.config.get("method") or "GET").upper()
    headers = dict(node.config.get("headers") or {})
    headers = {str(k): _render_http_string(str(v), variables) for k, v in headers.items()}
    if secret and secret.get("api_key"):
        header_name = secret.get("header_name") or "Authorization"
        value = secret["api_key"] if header_name != "Authorization" else f"Bearer {secret['api_key']}"
        headers.setdefault(header_name, value)

    body = _resolve_http_body(node.config.get("body"), variables)
    timeout = float(node.config.get("timeout_sec") or 30)

    # Prefer json= for dict/list; data= for raw strings
    kwargs = {"headers": headers, "transport": transport, "timeout": timeout}
    if body is not None and method not in ("GET", "HEAD"):
        if isinstance(body, (dict, list)):
            kwargs["json"] = body
        else:
            kwargs["content"] = str(body).encode("utf-8")

    # _http doesn't take timeout — call httpx directly for timeout support
    import httpx
    with httpx.Client(transport=transport, timeout=timeout) as client:
        response = client.request(method, url, headers=headers, **{k: v for k, v in kwargs.items() if k not in ("headers", "transport", "timeout")})
    if response.status_code >= 400 and node.config.get("fail_on_error", True):
        raise ConnectedExecutionError(
            f"HTTP {method} {url} returned {response.status_code}: {response.text[:500]}"
        )
    content_type = response.headers.get("content-type", "")
    try:
        parsed = response.json() if "application/json" in content_type else response.text
    except Exception:
        parsed = response.text
    out_key = node.config.get("output_key") or "api_response"
    return {
        out_key: parsed,
        "api_response": parsed,
        "api_status": response.status_code,
        "api_url": url,
        "api_method": method,
    }


def build_api_call_executor(*, credential_store: CredentialStore | None = None, transport=None) -> NodeExecutor:
    """HTTP Request / api_call: always performs a real request when URL is set.

    Optional tool binding mode=connected + credentialRef attaches an API key
    from the credential store. No binding still runs the HTTP call (public APIs).
    """
    store = credential_store or _default_credential_store

    def execute(node: AgentNode, spec: AgentSpecification, variables: dict[str, Any]) -> dict:
        url = node.config.get("url")
        if not url:
            return exec_generic_tool(node, spec, variables)
        secret = None
        if node.tool_refs:
            bindings_by_id = {t.id: t for t in spec.tools}
            binding = next((bindings_by_id[r] for r in node.tool_refs if r in bindings_by_id), None)
            if binding and binding.credential_ref and binding.mode == "connected":
                secret = _get_secret(store, spec, binding)
        return _connected_api_call(node, spec, variables, secret, transport=transport)

    return execute


# ---- read_file / write_file (real local filesystem, scoped) ---------------
# Needs no external credential and no network — same rationale
# database.py's SQLite executor gives for being "the one tool with a
# genuinely working implementation instead of a documented stub": there's
# no reason this should still return `"[stub file contents]"` when a real,
# safely-scoped local read/write is just as achievable in this sandbox as
# the SQLite database was.
#
# CSV/XLSX get a real second pass beyond "read the bytes": genuine
# tabular extraction via pandas — columns, row count, a preview, and
# per-numeric-column summary stats (count/mean/min/max/sum) — computed
# from whatever is actually in the file, not canned. This is what makes
# read_file the one tool a person can hand a real spreadsheet and get a
# genuine "extract info / summarize" result back without needing any
# external API, credential, or network access this sandbox doesn't have —
# see backend/real_executors/test_tools.py for a real .csv/.xlsx exercised
# end-to-end, and examples/sample_invoices.csv for something to try it on.

_TABULAR_EXTENSIONS = {".csv", ".xlsx", ".xls"}


def _resolve_scoped_path(base_dir: Path, relative_path: str) -> Path:
    """Relative paths are under base_dir. Absolute paths only if inside base_dir."""
    raw = Path(relative_path)
    base_resolved = base_dir.resolve()
    candidate = raw.resolve() if raw.is_absolute() else (base_dir / relative_path).resolve()
    if base_resolved not in candidate.parents and candidate != base_resolved:
        raise ConnectedExecutionError(
            f"Path '{relative_path}' escapes the agent's scoped file directory "
            f"({base_resolved}). Put the file under that directory, or set AGENT_FILES_DIR."
        )
    return candidate


def _write_tabular_file(path: Path, rows, columns=None) -> None:
    import pandas as pd
    if isinstance(rows, dict) and "table_preview" in rows:
        data = rows.get("table_preview") or rows.get("rows") or []
        cols = rows.get("table_columns") or columns
        df = pd.DataFrame(data, columns=cols) if cols else pd.DataFrame(data)
    elif isinstance(rows, dict) and "rows" in rows:
        df = pd.DataFrame(rows["rows"], columns=rows.get("columns") or columns)
    else:
        df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=False)
    elif suffix in {".xlsx", ".xls"}:
        df.to_excel(path, index=False)
    else:
        path.write_text(df.to_csv(index=False), encoding="utf-8")


def _read_tabular_file(path: Path, *, preview_rows: int = 5, max_rows: int = 500) -> dict:
    """Real pandas-backed CSV/XLSX extraction. Raises ConnectedExecutionError
    (not a silent empty result) if the file can't actually be parsed as a
    table — a malformed CSV or a password-protected workbook should fail
    loudly, the same "no silent stub fallback" principle as everything
    else in this module."""
    import pandas as pd

    try:
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        else:
            df = pd.read_excel(path)
    except Exception as exc:  # pandas raises many different exception types depending on the failure
        raise ConnectedExecutionError(f"Could not parse '{path.name}' as a table: {exc}") from exc

    if df.empty:
        return {
            "table_columns": list(df.columns),
            "table_row_count": 0,
            "table_preview": [],
            "table_summary": {},
            "rows": [],
            "table_truncated": False,
        }

    # Route every value through pandas' own JSON encoder (handles NaN ->
    # null and numpy int64/float64 -> plain JSON numbers) rather than
    # hand-rolling type coercion — the values landing in `variables` need
    # to survive json.dumps() for the trace/streaming layer regardless of
    # what numpy dtype pandas inferred.
    preview = json.loads(df.head(preview_rows).to_json(orient="records"))
    truncated = len(df) > max_rows
    rows = json.loads(df.head(max_rows).to_json(orient="records"))

    numeric_df = df.select_dtypes(include="number")
    table_summary: dict[str, dict[str, float]] = {}
    if not numeric_df.empty:
        described = json.loads(numeric_df.describe().to_json())
        for column, stats in described.items():
            table_summary[column] = {
                "count": stats.get("count"),
                "mean": stats.get("mean"),
                "min": stats.get("min"),
                "max": stats.get("max"),
                "sum": float(numeric_df[column].sum()),
            }

    return {
        "table_columns": [str(c) for c in df.columns],
        "table_row_count": int(len(df)),
        "table_preview": preview,
        "table_summary": table_summary,
        "rows": rows,
        "table_truncated": truncated,
    }


def build_read_file_executor(*, base_dir: str | None = None) -> NodeExecutor:
    base = Path(base_dir or os.environ.get("AGENT_FILES_DIR", "agent_platform_files"))
    base.mkdir(parents=True, exist_ok=True)

    def execute(node: AgentNode, spec: AgentSpecification, variables: dict[str, Any]) -> dict:
        binding = _try_resolve_binding(node, spec, "read")
        relative_path = node.config.get("path") or variables.get("file_path")
        # Local files need only a path — tool binding is optional.
        if binding is not None and binding.mode == "demo" and not relative_path:
            return exec_read_file(node, spec, variables)
        if not relative_path:
            raise ConnectedExecutionError(f"Node '{node.id}' read_file has no path configured.")
        path = _resolve_scoped_path(base, relative_path)
        if not path.exists():
            raise ConnectedExecutionError(f"File '{relative_path}' does not exist under the agent's file directory.")

        if path.suffix.lower() in _TABULAR_EXTENSIONS:
            result = _read_tabular_file(path)
            result["file_contents"] = None
            return result

        mime, _ = mimetypes.guess_type(str(path))
        if mime and not mime.startswith("text") and mime != "application/json":
            return {"file_contents": None, "file_bytes_base64": path.read_bytes().hex()}
        return {"file_contents": path.read_text(encoding="utf-8")}

    return execute


def build_write_file_executor(*, base_dir: str | None = None) -> NodeExecutor:
    base = Path(base_dir or os.environ.get("AGENT_FILES_DIR", "agent_platform_files"))
    base.mkdir(parents=True, exist_ok=True)

    def execute(node: AgentNode, spec: AgentSpecification, variables: dict[str, Any]) -> dict:
        binding = _try_resolve_binding(node, spec, "write")
        if binding is not None and binding.mode == "demo":
            return exec_write_file(node, spec, variables)
        relative_path = node.config.get("path") or variables.get("file_path")
        if not relative_path:
            raise ConnectedExecutionError(f"Node '{node.id}' is mode='connected' write_file but has no path configured.")
        # Prefer AI outputs over Read File leftovers in shared state.
        content_var = node.config.get("content_variable") or node.config.get("contentVariable")
        content = None
        if content_var:
            if content_var not in variables:
                raise ConnectedExecutionError(
                    f"Node '{node.id}' content_variable='{content_var}' not in variables "
                    f"(have: {sorted(variables.keys())}). Use 'ai_response' after Ask AI."
                )
            content = variables.get(content_var)
        if content is None or content == "":
            for key in ("ai_response", "draft_email", "summary", "file_contents"):
                if key in variables and variables[key] not in (None, ""):
                    content = variables[key]
                    break
        if content is None or content == "":
            if variables.get("table_preview") is not None:
                content = {
                    "table_preview": variables.get("table_preview"),
                    "table_columns": variables.get("table_columns"),
                    "rows": variables.get("table_preview"),
                    "columns": variables.get("table_columns"),
                }
            elif variables.get("rows") is not None:
                content = variables.get("rows")
            else:
                content = ""
        path = _resolve_scoped_path(base, relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        suffix = path.suffix.lower()
        if suffix in _TABULAR_EXTENSIONS and not isinstance(content, str):
            _write_tabular_file(path, content)
        elif suffix in _TABULAR_EXTENSIONS and isinstance(content, str):
            stripped = content.strip()
            if suffix == ".csv" and ("\n" in stripped and ("," in stripped or "\t" in stripped)):
                path.write_text(content, encoding="utf-8")
            else:
                _write_tabular_file(path, [{"response": content}], columns=["response"])
        else:
            path.write_text(str(content), encoding="utf-8")
        try:
            rel = str(path.relative_to(base.resolve()))
        except ValueError:
            rel = str(path)
        return {"file_written": True, "file_path": rel}

    return execute


def build_connected_tool_executors(
    *,
    credential_store: CredentialStore | None = None,
    transport: httpx.BaseTransport | None = None,
    database_demo_executor: NodeExecutor | None = None,
) -> dict:
    """The full set of mode-aware tool executors, ready to merge into
    compile_agent()'s `executor_overrides` alongside build_ai_executors()
    — see backend/api/app.py's get_executor_overrides().

    search_web is deliberately NOT here — it moved to
    backend/real_executors/search.py and is wired in get_executor_overrides()
    alongside the other Anthropic-backed AI executors, since a real
    search_web implementation needs an Anthropic client (server-side web
    search), the same dependency the AI executors already gate on, not a
    credential_store/transport the way the tools in this module do."""
    from backend.models.agent_spec import NodeType

    return {
        NodeType.send_email: build_send_email_executor(credential_store=credential_store, transport=transport),
        NodeType.read_email: build_read_email_executor(credential_store=credential_store, transport=transport),
        NodeType.database: build_database_connected_executor(
            credential_store=credential_store, transport=transport, demo_executor=database_demo_executor
        ),
        NodeType.api_call: build_api_call_executor(credential_store=credential_store, transport=transport),
        NodeType.read_file: build_read_file_executor(),
        NodeType.write_file: build_write_file_executor(),
        NodeType.code: __import__("backend.real_executors.code_exec", fromlist=["build_code_executor"]).build_code_executor(),
        NodeType.rag_index: __import__("backend.real_executors.rag", fromlist=["build_rag_index_executor"]).build_rag_index_executor(),
        NodeType.rag_query: __import__("backend.real_executors.rag", fromlist=["build_rag_query_executor"]).build_rag_query_executor(),
    }
