"""
The API layer: a thin FastAPI service exposing everything built so far —
mutation engine, compiler, NL compiler — as HTTP endpoints a frontend
(the canvas package, or anything else) can call.

    POST /agents                          create an agent from a spec
    GET  /agents/{id}                     fetch the current spec
    POST /agents/{id}/mutations           apply a mutation batch (canvas drag/connect/delete)
    POST /agents/{id}/instructions        NL instruction -> mutations (chat builder)
    POST /agents/{id}/compile             dry-run compile check ("Test" step, doc section 14)
    POST /agents/{id}/run                 compile + execute, pausing on human_approval
    POST /agents/{id}/resume              resume a paused run
    POST /agents/{id}/publish             promote current version to metadata.publishedVersion

Deliberately thin: every interesting decision (validation, atomicity,
version conflicts, graph compilation, tool permission checks) already
lives in backend/mutation/engine.py, backend/compiler/compiler.py, and
backend/nl_compiler/compiler.py. This file's only job is HTTP plumbing —
request parsing, status codes, and wiring to whatever store is injected.

The AgentStore itself is a FastAPI dependency (get_store), same pattern as
get_anthropic_client: the default resolves to the in-memory store (zero
setup, what every test in this repo except the Postgres persistence test
uses), and a real deployment overrides it with
backend.persistence.postgres_store.PostgresAgentStore — see
backend/persistence/test_postgres_persistence.py for exactly that override
in action, proving specs and paused runs survive across independent store
instances (i.e. a process restart) against a real Postgres database.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone

from anthropic import Anthropic
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from langgraph.types import Command
from pydantic import ValidationError as PydanticValidationError

from backend.api.schemas import (
    ChatApplyRequest,
    ChatProposalResponse,
    ChatProposeRequest,
    CompileCheckResponse,
    CreateAgentRequest,
    CreateWorkflowRequest,
    InstructionRequest,
    InstructionResponse,
    MutationsRequest,
    PublishResponse,
    RenameWorkflowRequest,
    ResumeRequest,
    RunRequest,
    RunResponse,
    WorkflowSummary,
)
from backend.api.store import AgentNotFoundError, AgentStore, store as _default_store
from backend.compiler.compiler import CompileError, compile_agent
from backend.credentials.models import CredentialCreateRequest, CredentialSummary
from backend.credentials.providers import CredentialValidationError, PROVIDER_REGISTRY
from backend.credentials.store import (
    CredentialAccessError,
    CredentialNotFoundError,
    CredentialStore,
    credential_store as _default_credential_store,
)
from backend.models.agent_spec import AgentSpecification, UpdateMetadataMutation, parse_mutation
from backend.mutation.engine import apply_mutations, publish as publish_spec
from backend.nl_compiler.compiler import NLCompileError, compile_instruction
from backend.tools.registry import public_registry

app = FastAPI(title="Agent Platform API", version="0.1.0")


# Lazily constructed, process-wide: if DATABASE_URL is set, get_store()
# builds (once) and reuses a real PostgresAgentStore instead of the
# in-memory default. Lazy so importing this module never requires
# Postgres to be reachable — only actually calling a route does, and only
# when DATABASE_URL is set. A module-level singleton (not "one per
# request") because PostgresAgentStore holds a live PostgresSaver whose
# setup() does real DDL — not something to redo on every request.
#
# NOTE: this block (and get_store below) deliberately sits right after
# app creation, before ANY route definition. It was previously defined
# much further down the file, after routes like list_agents() that
# reference get_store() as a Depends() default — since a `def`'s default
# argument values are evaluated the moment the `def` statement itself
# runs, that ordering raised `NameError: name 'get_store' is not defined`
# on every single import of this module (verified: `python -c "from
# backend.api.app import app"` failed on the unmodified file). Moving the
# dependency's definition above its first use is the actual fix — not a
# stylistic reshuffle.
_postgres_store: AgentStore | None = None


def get_store() -> AgentStore:
    """
    A FastAPI dependency so a real deployment can override this with
    PostgresAgentStore (backend/persistence/postgres_store.py) without
    touching any route handler below — see
    backend/persistence/test_postgres_persistence.py for that override in
    a test, and _resolve_default_store() below for how the *running
    server* picks it up automatically via DATABASE_URL.
    """
    return _resolve_default_store()


def _resolve_default_store() -> AgentStore:
    global _postgres_store
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        return _default_store
    if _postgres_store is None:
        from backend.persistence.postgres_store import PostgresAgentStore

        _postgres_store = PostgresAgentStore(database_url)
    return _postgres_store


@app.get("/tools/registry")
def tool_registry() -> list[dict]:
    """Expose the server's authoritative tool catalogue to clients."""
    return public_registry()


@app.get("/ai/providers")
def ai_providers() -> dict:
    """Public, non-secret status of AI provider configuration -- lets the
    inspector show which providers exist and which are actually usable
    right now, WITHOUT exposing anything secret: `hasCredentials` is a
    plain boolean (is the relevant API key env var present on this
    server), never the key's value, and `current` names the server's
    default provider (AI_PROVIDER env var), not a credential. This is
    exactly the "provider selection/credentials are server-side" boundary
    from the spec: the frontend can see and choose FROM this list, but
    every credential this reflects lives only in this process's own
    environment.

    `models` is the curated, per-provider model list (backend/ai_providers/
    registry.py's KNOWN_MODELS) -- lets the Ask AI inspector's model
    dropdown populate itself from the server's own truth instead of
    hardcoding model names a second time in the frontend, and gives it
    something to auto-select from the instant a provider is switched
    (see AiProviderSelector.tsx / NodeInspector.tsx's provider-change
    handler for where "switching providers must automatically replace an
    incompatible model" actually happens).
    """
    from backend.ai_providers.registry import (
        KNOWN_MODELS,
        PLANNED_PROVIDERS,
        SUPPORTED_PROVIDERS,
        has_credentials,
        resolve_provider_name,
    )

    return {
        "current": resolve_provider_name(),
        "providers": [
            {
                "id": name,
                "status": "available",
                "hasCredentials": has_credentials(name),
                "models": KNOWN_MODELS.get(name, []),
            }
            for name in SUPPORTED_PROVIDERS
        ]
        + [
            {"id": name, "status": "coming_soon", "hasCredentials": False, "models": []}
            for name in PLANNED_PROVIDERS
        ],
    }


def get_credential_store() -> CredentialStore:
    """FastAPI dependency, same shape as get_store — the default resolves
    to the process-wide in-memory singleton; a real deployment overrides
    this with a Postgres-backed implementation the same way get_store is
    overridden, without touching any route below."""
    return _default_credential_store


def get_owner_id(request: Request) -> str:
    """Resolves the acting principal for credential ownership checks.

    There is no real authentication yet (P0.4, not built) — this reads an
    `X-User-Id` header if a caller sends one and falls back to a fixed
    'default' owner otherwise, which is what every route implicitly
    assumed before this module existed anyway. This function is the one
    seam that needs to change (to pull the principal off a verified
    session/JWT instead of a trusted client header) once real auth lands;
    every route below already depends on it rather than hardcoding
    'default' inline, so that change is localized to this one function.
    """
    return request.headers.get("X-User-Id") or "default"


@app.get("/credentials/providers")
def list_credential_providers() -> list[dict]:
    """Expose the provider catalogue (required fields, auth type) so a
    'Connect' UI knows what form to render for each provider, without
    hardcoding that shape on the frontend."""
    return [
        {
            "id": p.id,
            "name": p.name,
            "authType": p.auth_type,
            "fields": [
                {"name": f.name, "required": f.required, "description": f.description}
                for f in p.fields
            ],
        }
        for p in PROVIDER_REGISTRY.values()
    ]




# ---- Agent files (upload into AGENT_FILES_DIR for Read File / RAG) --------

def _agent_files_root() -> Path:
    from pathlib import Path as _P
    root = _P(os.environ.get("AGENT_FILES_DIR", "agent_platform_files"))
    root.mkdir(parents=True, exist_ok=True)
    return root


@app.get("/files")
def list_agent_files() -> dict:
    """List files available under AGENT_FILES_DIR (non-recursive top level + one level)."""
    root = _agent_files_root()
    files = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if ".rag" in p.parts:
            continue
        try:
            rel = str(p.relative_to(root))
        except ValueError:
            continue
        files.append({"path": rel, "size": p.stat().st_size, "suffix": p.suffix.lower()})
    return {"directory": str(root.resolve()), "files": files}


@app.post("/files/upload")
async def upload_agent_file(file: UploadFile = File(...), subdir: str = "") -> dict:
    """Upload a file into AGENT_FILES_DIR so Read File / RAG can use it by relative path."""
    from pathlib import Path as _P
    import re as _re

    root = _agent_files_root()
    name = _P(file.filename or "upload.bin").name
    name = _re.sub(r"[^A-Za-z0-9._\- ]+", "_", name).strip() or "upload.bin"
    dest_dir = root
    if subdir:
        # Allow only simple relative segments (no .. or drive letters)
        safe = _re.sub(r"[^A-Za-z0-9._/\-]+", "_", subdir)
        safe = safe.replace("\\", "/").strip("/")
        while ".." in safe.split("/"):
            safe = "/".join(part for part in safe.split("/") if part not in ("", ".."))
        dest_dir = root / safe
        dest_dir.mkdir(parents=True, exist_ok=True)
        try:
            dest_dir.resolve().relative_to(root.resolve())
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid subdir")
    dest = dest_dir / name
    data = await file.read()
    dest.write_bytes(data)
    rel = str(dest.relative_to(root))
    return {"ok": True, "path": rel, "size": len(data), "directory": str(root.resolve())}


@app.post("/credentials", response_model=CredentialSummary, status_code=201)
def create_credential(
    body: CredentialCreateRequest,
    store: CredentialStore = Depends(get_credential_store),
    owner_id: str = Depends(get_owner_id),
) -> CredentialSummary:
    """Stores a new credential. The request body's `secret` field is never
    echoed back — the response is a CredentialSummary, which structurally
    cannot carry secret material (see credentials/models.py)."""
    try:
        return store.create(
            provider=body.provider,
            name=body.name,
            secret=body.secret,
            owner_id=body.owner_id or owner_id,
        )
    except CredentialValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.get("/credentials", response_model=list[CredentialSummary])
def list_credentials(
    store: CredentialStore = Depends(get_credential_store),
    owner_id: str = Depends(get_owner_id),
) -> list[CredentialSummary]:
    return store.list(owner_id)


@app.get("/credentials/{credential_id}", response_model=CredentialSummary)
def get_credential(
    credential_id: str,
    store: CredentialStore = Depends(get_credential_store),
    owner_id: str = Depends(get_owner_id),
) -> CredentialSummary:
    try:
        return store.get_summary(credential_id, owner_id)
    except (CredentialNotFoundError, CredentialAccessError):
        raise HTTPException(status_code=404, detail=f"No credential with id '{credential_id}'.")


@app.delete("/credentials/{credential_id}", status_code=204)
def delete_credential(
    credential_id: str,
    store: CredentialStore = Depends(get_credential_store),
    owner_id: str = Depends(get_owner_id),
) -> None:
    try:
        store.delete(credential_id, owner_id)
    except (CredentialNotFoundError, CredentialAccessError):
        raise HTTPException(status_code=404, detail=f"No credential with id '{credential_id}'.")


@app.post("/credentials/{credential_id}/validate", response_model=CredentialSummary)
def validate_credential(
    credential_id: str,
    store: CredentialStore = Depends(get_credential_store),
    owner_id: str = Depends(get_owner_id),
) -> CredentialSummary:
    """Re-checks the stored secret's shape and records the outcome
    (`status`, `last_validated_at`) on the credential. See
    credentials/providers.py for exactly what 'validation' means without
    live egress to the provider's API."""
    try:
        store.validate_connection(credential_id, owner_id)
        return store.get_summary(credential_id, owner_id)
    except (CredentialNotFoundError, CredentialAccessError):
        raise HTTPException(status_code=404, detail=f"No credential with id '{credential_id}'.")



from pydantic import BaseModel as _PydanticBaseModel


class SmtpCredentialRequest(_PydanticBaseModel):
    email: str
    password: str
    host: str = "smtp.gmail.com"
    port: int = 587
    name: str = "My email (SMTP)"
    from_email: str | None = None
    use_tls: bool = True


@app.post("/credentials/smtp", response_model=CredentialSummary)
def create_smtp_credential(
    body: SmtpCredentialRequest,
    store: CredentialStore = Depends(get_credential_store),
    owner_id: str = Depends(get_owner_id),
) -> CredentialSummary:
    """Upsert SMTP settings: same email (or the only existing SMTP cred) is
    updated in place so repeated Save does not mint a new cred_ id."""
    secret = {
        "host": body.host,
        "port": str(body.port),
        "username": body.email,
        "password": body.password,
        "from_email": body.from_email or body.email,
        "use_tls": "true" if body.use_tls else "false",
    }
    try:
        existing = [c for c in store.list(owner_id) if c.provider == "smtp"]
        match = None
        for c in existing:
            try:
                s = store.get_secret(c.id, owner_id)
            except Exception:
                continue
            if (s.get("username") or "").lower() == body.email.strip().lower():
                match = c
                break
        if match is None and len(existing) == 1:
            match = existing[0]
        if match is not None:
            return store.update_secret(match.id, owner_id, secret)
        return store.create(
            provider="smtp",
            name=body.name,
            secret=secret,
            owner_id=owner_id,
        )
    except CredentialValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))



@app.get("/credentials/oauth/gmail/config")
def gmail_oauth_config_get() -> dict:
    from backend.credentials.gmail_oauth import load_oauth_app_config
    cfg = load_oauth_app_config()
    # Never return the full secret to the browser — only whether it is set.
    return {
        "client_id": cfg.get("client_id") or "",
        "client_secret_set": bool(cfg.get("client_secret")),
        "redirect_uri": cfg.get("redirect_uri") or "",
        "configured": bool(cfg.get("client_id") and cfg.get("client_secret")),
    }


@app.post("/credentials/oauth/gmail/config")
def gmail_oauth_config_save(body: dict) -> dict:
    from backend.credentials.gmail_oauth import GmailOAuthError, save_oauth_app_config
    try:
        saved = save_oauth_app_config(
            client_id=str(body.get("client_id") or ""),
            client_secret=str(body.get("client_secret") or ""),
            redirect_uri=str(body.get("redirect_uri") or "") or None,
        )
    except GmailOAuthError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "ok": True,
        "client_id": saved["client_id"],
        "redirect_uri": saved["redirect_uri"],
        "client_secret_set": True,
        "configured": True,
    }


@app.get("/credentials/oauth/gmail/start")

def gmail_oauth_start(
    name: str = "Gmail",
    owner_id: str = Depends(get_owner_id),
) -> dict:
    """Begin Gmail OAuth. Returns {authorize_url, state}. Open authorize_url in a browser."""
    from backend.credentials.gmail_oauth import GmailOAuthError, build_authorize_url

    try:
        url, state = build_authorize_url(owner_id=owner_id, credential_name=name)
    except GmailOAuthError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return {"authorize_url": url, "state": state, "redirect_uri": __import__("backend.credentials.gmail_oauth", fromlist=["redirect_uri"]).redirect_uri()}


@app.get("/credentials/oauth/gmail/callback")
def gmail_oauth_callback(
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
    store: CredentialStore = Depends(get_credential_store),
):
    """Google redirects here after consent. Stores the Gmail credential and shows a simple success page."""
    from fastapi.responses import HTMLResponse
    from backend.credentials.gmail_oauth import (
        GmailOAuthError,
        exchange_code_for_tokens,
        pop_state,
    )

    if error:
        return HTMLResponse(
            f"<h2>Gmail connect failed</h2><p>{error}</p><p>You can close this window.</p>",
            status_code=400,
        )
    if not code or not state:
        return HTMLResponse("<h2>Missing code/state</h2>", status_code=400)
    try:
        meta = pop_state(state)
        tokens = exchange_code_for_tokens(code)
        secret = {
            "access_token": tokens["access_token"],
            "refresh_token": tokens.get("refresh_token") or "",
            "client_id": tokens["client_id"],
            "client_secret": tokens["client_secret"],
        }
        summary = store.create(
            provider="gmail",
            name=meta.get("credential_name") or "Gmail",
            secret=secret,
            owner_id=meta.get("owner_id") or "default",
        )
    except (GmailOAuthError, CredentialValidationError) as exc:
        return HTMLResponse(
            f"<h2>Gmail connect failed</h2><p>{exc}</p><p>You can close this window.</p>",
            status_code=400,
        )
    html = f"""
    <html><body style="font-family: system-ui, sans-serif; padding: 2rem; max-width: 36rem;">
      <h2 style="color:#065f46;">Gmail connected</h2>
      <p>Use this credential id on <strong>Send Email</strong> / <strong>Read Email</strong>
         → Connection reference (mode = connected).</p>
      <div style="display:flex; gap:8px; align-items:center; padding:12px; background:#ecfdf5; border:1px solid #a7f3d0; border-radius:8px;">
        <code id="cred" style="flex:1; word-break:break-all; font-size:14px;">{summary.id}</code>
        <button type="button" onclick="navigator.clipboard.writeText(document.getElementById('cred').innerText); this.innerText='Copied';"
          style="padding:8px 12px; border-radius:6px; border:1px solid #6ee7b7; background:#fff; cursor:pointer;">Copy</button>
      </div>
      <p style="color:#6b7280; font-size:13px;">You can close this window after copying.</p>
    </body></html>
    """
    return HTMLResponse(html)


@app.get("/agents", response_model=list[AgentSpecification])
def list_agents(store: AgentStore = Depends(get_store)) -> list[AgentSpecification]:
    return store.list_agents()

# CORS: the canvas app (packages/app, typically http://localhost:5173) and
# the API (typically http://localhost:8000) are different origins, so the
# browser enforces CORS even for plain same-machine local dev — this is
# NOT optional for the app shell to work, only invisible if you test
# exclusively via curl/Swagger (served from the API's own origin, so
# same-origin, so CORS never applies). Configurable via CORS_ALLOWED_ORIGINS
# (comma-separated) rather than a hardcoded wildcard, so a real deployment
# doesn't inherit a dev-only "allow everything" default; unset falls back
# to the common local Vite dev ports.
_default_dev_origins = ["http://localhost:5173", "http://127.0.0.1:5173"]
_cors_origins_env = os.environ.get("CORS_ALLOWED_ORIGINS")
_cors_origins = [o.strip() for o in _cors_origins_env.split(",") if o.strip()] if _cors_origins_env else _default_dev_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _ensure_cors_on_errors(request: Request, call_next):
    """
    Belt-and-suspenders fix for a real, confirmed Starlette/FastAPI gotcha:
    a genuinely UNHANDLED exception is caught by Starlette's
    ServerErrorMiddleware, which is UNCONDITIONALLY the outermost layer —
    outside CORSMiddleware, regardless of add_middleware() ordering, and
    outside even a registered `@app.exception_handler(Exception)` handler
    (those attach to ServerErrorMiddleware too, not the inner
    ExceptionMiddleware that CORSMiddleware actually wraps). The result: a
    bare 500 with no Access-Control-Allow-Origin header, which a browser
    making a cross-origin request (the app shell at :5173 calling the API
    at :8000) can't read at all — it surfaces in JS as a generic
    "TypeError: Failed to fetch" with zero indication anything even
    reached the server, or what actually went wrong.

    Verified this concretely before writing the fix: hit /instructions
    with no ANTHROPIC_API_KEY configured (a real, easy-to-hit unhandled
    exception — the Anthropic client complains it has no auth method) and
    inspected the raw response headers. Confirmed missing. Then confirmed
    a bare `@app.exception_handler(Exception)` handler ALSO didn't fix
    it — same missing header, despite the response body being clean JSON.
    Catching the exception in OUR OWN middleware, which genuinely sits
    inside the ASGI call chain we control, and attaching the header
    ourselves (rather than hoping something downstream will) is what
    actually closes this — re-verified with the same raw-header check
    after this fix, see backend/api/test_error_cors.py.
    """
    origin = request.headers.get("origin")
    cors_headers = {}
    if origin and (origin in _cors_origins or "*" in _cors_origins):
        cors_headers = {
            "Access-Control-Allow-Origin": origin,
            "Access-Control-Allow-Credentials": "true",
            "Vary": "Origin",
        }
    try:
        return await call_next(request)
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"detail": f"Internal server error: {exc}"},
            headers=cors_headers,
        )


def get_anthropic_client() -> Anthropic:
    """
    A FastAPI dependency so tests can override this with a fake client
    (see backend/api/test_app.py) without needing a real API key —
    overriding this dependency REPLACES this function's body entirely, so
    the ANTHROPIC_API_KEY check below only ever runs on the real,
    non-overridden path.

    Checked here rather than inline in apply_agent_instruction specifically
    so it can't accidentally break that override pattern: an earlier draft
    of this check lived in the route itself, which meant it fired even for
    requests using a fake test client, since it never looked at `client`
    at all — just the environment. Moving it here means "was the
    dependency overridden" is exactly "did this function's body even run,"
    which is the correct condition.

    Raises a clean, actionable 503 instead of letting a bare
    "Could not resolve authentication method" Anthropic SDK exception
    propagate to the catch-all error middleware and read as a generic
    platform bug rather than "you haven't configured this yet" (V9's "Fix 4").
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail={
                "error": "AI Builder unavailable — no ANTHROPIC_API_KEY is configured on the server.",
                "reason": "no_api_key",
            },
        )
    return Anthropic()


_executor_overrides: dict | None = None
_warned_no_api_key = False
_warned_no_search_api_key = False


def get_executor_overrides() -> dict:
    """
    A FastAPI dependency, same pattern as get_store/get_anthropic_client:
    the default lazily resolves to REAL executors (backend/real_executors/)
    -- classify/extract_information/summarize/make_decision/ai_agent/ask_ai
    actually call an LLM (Anthropic or OpenAI, whichever AI_PROVIDER
    selects -- see backend/ai_providers/registry.py) when that provider's
    API key is set, `database` writes to real SQLite, `search_web` makes a
    real web search via Anthropic's server-side search tool (see
    backend/real_executors/search.py), and -- see
    backend/real_executors/tools.py -- send_email/read_email/database/
    api_call/read_file/write_file are all mode-aware: a node whose
    ToolBinding is mode="connected" with a real credential
    (backend/credentials/) genuinely calls out (Gmail, Sheets, or an
    arbitrary HTTP endpoint) instead of returning the same canned
    `{"email_sent": True}` a mode="demo" binding would.

    AI execution (and search_web, which depends on Anthropic specifically
    regardless of AI_PROVIDER) falls back to stub executors when the
    relevant API key isn't configured, WITH an explicit one-time warning
    per missing key, rather than either (a) crashing the whole app at
    startup or every run, or (b) silently using fake stub data with no
    indication anything's off. Tool execution (send_email, database, etc.)
    doesn't need that fallback -- it's already mode-aware per binding, so a
    demo-mode binding behaves identically with or without an API key, and
    a connected-mode binding either has a real credential or fails loudly
    at compile time (P0.1) or run time (real_executors/tools.py), never
    silently.
    """
    global _executor_overrides, _warned_no_api_key, _warned_no_search_api_key
    if _executor_overrides is None:
        from backend.ai_providers.registry import build_provider, has_credentials, resolve_provider_name
        from backend.real_executors.database import build_database_executor
        from backend.real_executors.tools import build_connected_tool_executors

        db_path = os.environ.get("AGENT_DB_SQLITE_PATH", "agent_platform_data.sqlite3")
        # Real SQLite backs `database` in demo mode too -- same rationale
        # as before this function existed: SQLite needs no external
        # credential this sandbox can't provide, so there's no reason a
        # demo-mode database node should use a fake in-memory stub instead.
        database_demo_executor = build_database_executor(db_path)
        tool_overrides = build_connected_tool_executors(database_demo_executor=database_demo_executor)

        # AI_PROVIDER selects which provider every AI-category node uses
        # (default: "anthropic", preserving every existing deployment's
        # behavior unchanged). This is a server-only environment variable
        # -- it is never read from, or exposed to, a request or an
        # AgentSpec. See backend/ai_providers/registry.py.
        provider_name = resolve_provider_name()
        if not has_credentials(provider_name):
            if not _warned_no_api_key:
                from backend.ai_providers.registry import api_key_env_var

                print(
                    f"[agent-platform] WARNING: {api_key_env_var(provider_name)} is not set "
                    f"(AI_PROVIDER='{provider_name}'). AI nodes (classify, extract_information, "
                    "summarize, make_decision, ai_agent, ask_ai) will run with STUB executors, "
                    f"not real {provider_name} calls. Set {api_key_env_var(provider_name)} to "
                    "enable real AI execution."
                )
                _warned_no_api_key = True
            ai_overrides: dict = {}
        else:
            from backend.real_executors.ai import build_ai_executors

            ai_overrides = build_ai_executors(build_provider(provider_name))

        # search_web is Anthropic-specific (server-side web search tool)
        # regardless of AI_PROVIDER -- checked against ANTHROPIC_API_KEY
        # directly, independent of whichever provider the other AI nodes
        # are using.
        from backend.models.agent_spec import NodeType
        from backend.real_executors.search import build_search_web_executor_with_fallback

        if not os.environ.get("ANTHROPIC_API_KEY"):
            if not _warned_no_search_api_key:
                print(
                    "[agent-platform] WARNING: ANTHROPIC_API_KEY is not set. search_web will "
                    "run with a STUB executor (real web search needs Anthropic's server-side "
                    "search tool regardless of AI_PROVIDER). Set ANTHROPIC_API_KEY to enable "
                    "real web search."
                )
                _warned_no_search_api_key = True
            search_overrides = {NodeType.search_web: build_search_web_executor_with_fallback(None)}
        else:
            from anthropic import Anthropic

            search_overrides = {NodeType.search_web: build_search_web_executor_with_fallback(Anthropic())}

        _executor_overrides = {**tool_overrides, **ai_overrides, **search_overrides}
    return _executor_overrides


def _apply_spec_provider_override(spec: "AgentSpecification", executor_overrides: dict) -> dict:
    """`get_executor_overrides()` above resolves ONE server-wide default
    provider (AI_PROVIDER) and caches it for the whole process, since
    constructing a provider client on every request would be wasteful for
    the common case where every agent uses the server's default.

    But `spec.runtimeConfig.model` (validated in
    backend/models/agent_spec.py's RuntimeConfig) lets an individual
    AgentSpecification ask for a SPECIFIC provider via the
    "<provider>:<model>" convention — e.g. an agent explicitly authored
    to use "openai:gpt-4o" should use OpenAI even if the server's default
    AI_PROVIDER is "anthropic". This function is what actually honors
    that per-spec request: called after a spec is loaded and the cached
    default overrides are resolved, it builds a fresh, uncached AI
    provider ONLY when the spec asks for something different from the
    default, falling back to the cached default (already warned about)
    if that provider's credentials aren't configured — this field is a
    spec-wide PREFERENCE, not the strict per-node override feature (see
    compiler.py's _check_ai_provider_availability for where the strict,
    never-silently-fall-back behavior actually lives: per-node
    config.provider, set by the Ask AI inspector).

    Provider selection stays entirely server-side either way: this
    function reads spec.runtimeConfig.model (a public, non-secret
    provider *name*), and the concrete client construction
    (build_provider -> Anthropic()/OpenAI()) still resolves its API key
    from this process's own environment, never from the spec or request.
    """
    if not spec.runtime_config or not spec.runtime_config.model:
        return executor_overrides

    from backend.ai_providers.registry import (
        build_provider,
        has_credentials,
        parse_model_string,
        resolve_provider_name,
    )

    requested_provider, requested_model = parse_model_string(spec.runtime_config.model)
    if requested_provider is None:
        return executor_overrides  # bare model name, no provider override requested

    default_provider = resolve_provider_name()
    if requested_provider == default_provider:
        return executor_overrides  # already what the cached default provides

    if not has_credentials(requested_provider):
        # This agent's runtimeConfig.model NAMES this provider, but that's
        # a spec-wide preference field that predates the strict per-node
        # override feature (see compiler.py's
        # _check_ai_provider_availability's docstring for why per-node
        # config.provider is strict but this field isn't) — falls back to
        # the server's default (already covered by the startup warning)
        # rather than failing every run over a preference this
        # deployment can't currently satisfy.
        return executor_overrides

    from backend.real_executors.ai import build_ai_executors

    overridden_ai = build_ai_executors(build_provider(requested_provider, requested_model))
    return {**executor_overrides, **overridden_ai}


def _resolve_executor_overrides(spec: "AgentSpecification", executor_overrides: dict) -> dict:
    """Thin wrapper around _apply_spec_provider_override for the 4 route
    handlers below: converts its ProviderConfigurationError into the same
    422-with-node_id shape a CompileError already gets, so a client sees
    one consistent "spec/config problem" error format regardless of
    whether the problem is a bad tool binding or a missing provider key."""
    from backend.ai_providers.registry import ProviderConfigurationError

    try:
        return _apply_spec_provider_override(spec, executor_overrides)
    except ProviderConfigurationError as e:
        raise HTTPException(status_code=422, detail={"error": str(e), "node_id": None})


def _get_spec_or_404(store: AgentStore, agent_id: str) -> AgentSpecification:
    try:
        return store.get(agent_id)
    except AgentNotFoundError:
        raise HTTPException(status_code=404, detail=f"No agent with id '{agent_id}'.")


@app.post("/agents", response_model=AgentSpecification, status_code=201)
def create_agent(
    body: CreateAgentRequest,
    store: AgentStore = Depends(get_store),
    owner_id: str = Depends(get_owner_id),
) -> AgentSpecification:
    if store.exists(body.spec.metadata.id):
        raise HTTPException(status_code=409, detail=f"Agent '{body.spec.metadata.id}' already exists.")
    # The creating request's own identity (get_owner_id — X-User-Id today,
    # a real authenticated principal once auth exists) becomes the
    # agent's owner, unconditionally overriding whatever `metadata.owner`
    # the client submitted in the request body. Before this, `metadata.owner`
    # was just client-supplied, unauthenticated data that
    # compiler.py's _check_credential_refs then TRUSTED as an
    # authorization boundary for which credentials a spec could use — a
    # client could submit `{"metadata": {"owner": "alice"}}` in the
    # create payload and immediately gain compile-time access to
    # everything alice owns, having never authenticated as alice at all.
    # Tying it to the same identity mechanism credential ownership
    # already uses closes that gap without needing a real auth system to
    # exist yet — see get_owner_id's own docstring for the single seam
    # that changes when one does.
    spec_dict = body.spec.model_dump(by_alias=True)
    spec_dict["metadata"]["owner"] = owner_id
    spec = AgentSpecification.model_validate(spec_dict)
    store.create(spec)
    return spec


# ---- Multi-workflow: workflow CRUD ----------------------------------
#
# A "workflow" is exactly what every route above already calls an
# "agent" (one AgentSpecification, keyed by its own metadata.id in
# AgentStore) — see AgentMetadata.agent_id's docstring in
# backend/models/agent_spec.py for the full reasoning. These routes are
# what let ONE Agent (a lightweight grouping identified by `agent_id`,
# with no model of its own — it's just the value shared across several
# workflows' metadata.agentId) contain several independent workflows.
# Every read/write below still goes through the exact same AgentStore,
# compiler, mutation engine, and credential system every other route
# uses — nothing here duplicates that machinery.


def _to_workflow_summary(spec: AgentSpecification) -> WorkflowSummary:
    return WorkflowSummary(
        workflow_id=spec.metadata.id,
        agent_id=spec.metadata.effective_agent_id,
        name=spec.metadata.name,
        description=spec.metadata.description,
        enabled=spec.metadata.enabled,
        status=spec.metadata.status.value,
        version=spec.metadata.version,
        published_version=spec.metadata.published_version,
        updated_at=spec.metadata.updated_at.isoformat(),
    )


@app.post("/agents/{agent_id}/workflows", response_model=AgentSpecification, status_code=201)
def create_workflow(
    agent_id: str,
    body: CreateWorkflowRequest,
    store: AgentStore = Depends(get_store),
    owner_id: str = Depends(get_owner_id),
) -> AgentSpecification:
    """Creates a new, independent workflow under `agent_id`. Independent
    means genuinely independent: its own workflowId (== metadata.id, so
    it gets its own AgentStore entry — its own draft state, its own
    version history, its own checkpointer for run state — automatically,
    the same way every existing single-workflow agent already does),
    starting from a minimal one-node (manual trigger) graph a person can
    build out from scratch on the canvas or via the chat builder.
    """
    workflow_id = f"agt_{uuid.uuid4().hex[:12]}"
    now = datetime.now(timezone.utc).isoformat()
    spec = AgentSpecification.model_validate(
        {
            "metadata": {
                "id": workflow_id,
                "agentId": agent_id,
                "name": body.name,
                "description": body.description,
                "status": "draft",
                "version": "0.0.1",
                "createdAt": now,
                "updatedAt": now,
                "owner": owner_id,
                "enabled": True,
            },
            "goal": body.description or body.name,
            "nodes": [
                {"id": "n_trigger", "type": "trigger_manual", "category": "trigger", "label": "Start"},
            ],
            "edges": [],
        }
    )
    store.create(spec)
    return spec


@app.get("/agents/{agent_id}/workflows", response_model=list[WorkflowSummary])
def list_workflows(agent_id: str, store: AgentStore = Depends(get_store)) -> list[WorkflowSummary]:
    """Lists every workflow belonging to `agent_id` — including, for a
    legacy single-workflow agent, itself (AgentMetadata.effective_agent_id
    falls back to the workflow's own id when metadata.agentId was never
    set), so an old agent created before this feature existed shows up
    here as a one-workflow list with no migration needed."""
    return [_to_workflow_summary(s) for s in store.list_by_agent(agent_id)]


def _get_workflow_or_404(store: AgentStore, agent_id: str, workflow_id: str) -> AgentSpecification:
    spec = _get_spec_or_404(store, workflow_id)
    if spec.metadata.effective_agent_id != agent_id:
        # Same "don't distinguish wrong-owner from nonexistent" reasoning
        # credentials/store.py already uses — a workflow that exists but
        # belongs to a DIFFERENT agent shouldn't be distinguishable via
        # this URL from one that doesn't exist at all.
        raise HTTPException(status_code=404, detail=f"No workflow '{workflow_id}' under agent '{agent_id}'.")
    return spec


@app.get("/agents/{agent_id}/workflows/{workflow_id}", response_model=AgentSpecification)
def get_workflow(agent_id: str, workflow_id: str, store: AgentStore = Depends(get_store)) -> AgentSpecification:
    return _get_workflow_or_404(store, agent_id, workflow_id)


@app.post("/agents/{agent_id}/workflows/{workflow_id}/rename", response_model=AgentSpecification)
def rename_workflow(
    agent_id: str,
    workflow_id: str,
    body: RenameWorkflowRequest,
    store: AgentStore = Depends(get_store),
) -> AgentSpecification:
    """Renames a workflow. Reuses the existing update_metadata mutation
    and the SAME apply_mutations() + compare_and_set() path every other
    write in this file goes through — not a parallel rename
    implementation — so a rename gets exactly the same version-conflict
    and validation handling as any other edit."""
    spec = _get_workflow_or_404(store, agent_id, workflow_id)
    mutation = UpdateMetadataMutation(patch={"name": body.name})
    result = apply_mutations(spec, [mutation], expected_version=spec.metadata.version)
    if not result.ok or result.spec is None:
        raise HTTPException(status_code=422, detail=[err.model_dump() for err in (result.errors or [])])
    if not store.compare_and_set(workflow_id, spec.metadata.version, result.spec):
        raise HTTPException(status_code=409, detail="Version conflict: this workflow changed underneath the rename. Re-fetch and retry.")
    return result.spec


@app.post("/agents/{agent_id}/workflows/{workflow_id}/duplicate", response_model=AgentSpecification, status_code=201)
def duplicate_workflow(
    agent_id: str,
    workflow_id: str,
    store: AgentStore = Depends(get_store),
    owner_id: str = Depends(get_owner_id),
) -> AgentSpecification:
    """A full, independent copy: new workflowId, fresh draft status and
    version history (does NOT copy the source's version history or
    published snapshot — the duplicate starts as its own draft, exactly
    like a brand-new workflow would, just pre-populated with the
    source's current nodes/edges/tools/policies)."""
    source = _get_workflow_or_404(store, agent_id, workflow_id)
    new_id = f"agt_{uuid.uuid4().hex[:12]}"
    now = datetime.now(timezone.utc).isoformat()
    spec_dict = source.model_dump(by_alias=True)
    spec_dict["metadata"] = {
        **spec_dict["metadata"],
        "id": new_id,
        "agentId": agent_id,
        "name": f"{source.metadata.name} (copy)",
        "status": "draft",
        "version": "0.0.1",
        "publishedVersion": None,
        "createdAt": now,
        "updatedAt": now,
        "owner": owner_id,
    }
    new_spec = AgentSpecification.model_validate(spec_dict)
    store.create(new_spec)
    return new_spec


@app.delete("/agents/{agent_id}/workflows/{workflow_id}", status_code=204)
def delete_workflow(agent_id: str, workflow_id: str, store: AgentStore = Depends(get_store)) -> None:
    _get_workflow_or_404(store, agent_id, workflow_id)  # 404s if it doesn't exist or belongs to a different agent
    store.delete(workflow_id)


@app.get("/agents/{agent_id}", response_model=AgentSpecification)
def get_agent(agent_id: str, store: AgentStore = Depends(get_store)) -> AgentSpecification:
    return _get_spec_or_404(store, agent_id)


@app.get("/agents/{agent_id}/versions", response_model=list[AgentSpecification])
def list_agent_versions(agent_id: str, store: AgentStore = Depends(get_store)) -> list[AgentSpecification]:
    try:
        return store.list_versions(agent_id)
    except AgentNotFoundError:
        raise HTTPException(status_code=404, detail=f"No versions for agent '{agent_id}'.")


@app.get("/agents/{agent_id}/versions/{version}", response_model=AgentSpecification)
def get_agent_version(agent_id: str, version: str, store: AgentStore = Depends(get_store)) -> AgentSpecification:
    try:
        return store.get_version(agent_id, version)
    except AgentNotFoundError:
        raise HTTPException(status_code=404, detail=f"No version '{version}' for agent '{agent_id}'.")


@app.get("/agents/{agent_id}/published", response_model=AgentSpecification)
def get_published_agent(agent_id: str, store: AgentStore = Depends(get_store)) -> AgentSpecification:
    try:
        return store.get_published(agent_id)
    except AgentNotFoundError:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' has no published version.")


@app.post("/agents/{agent_id}/mutations")
def apply_agent_mutations(
    agent_id: str,
    body: MutationsRequest,
    store: AgentStore = Depends(get_store),
    owner_id: str = Depends(get_owner_id),
):
    spec = _get_spec_or_404(store, agent_id)

    try:
        mutations = [parse_mutation(m) for m in body.mutations]
    except PydanticValidationError as e:
        raise HTTPException(status_code=422, detail=f"Malformed mutation: {e}")

    # Ownership is an authorization boundary (compiler.py's
    # _check_credential_refs trusts spec.metadata.owner to decide which
    # credentials a spec may use) — it must not be reassignable through a
    # generic metadata patch, or any request carrying an X-User-Id header
    # could hand itself access to another owner's credentials by mutating
    # `metadata.owner` after the fact. A real ownership TRANSFER is a
    # distinct, deliberate operation this API doesn't offer yet — not
    # something to allow as a side effect of an unrelated metadata edit.
    for mutation in mutations:
        if mutation.kind == "update_metadata" and "owner" in mutation.patch:
            raise HTTPException(
                status_code=422,
                detail="Cannot change an agent's owner via update_metadata — ownership is not transferable through this endpoint.",
            )

    result = apply_mutations(spec, mutations, expected_version=body.expected_version)

    if not result.ok:
        is_conflict = any(err.path == "metadata.version" for err in (result.errors or []))
        raise HTTPException(
            status_code=409 if is_conflict else 422,
            detail=[err.model_dump() for err in (result.errors or [])],
        )

    # apply_mutations() already checked `body.expected_version` against the
    # `spec` object this request fetched — but that's a Python-level check
    # against a snapshot, not atomic with the write. Two concurrent
    # requests can both pass it against the same starting version and both
    # reach this point. compare_and_set() is the real guard: it re-checks
    # the version AS PART OF the write itself (atomically, at the store),
    # so only one of two racing requests can actually succeed.
    if not store.compare_and_set(agent_id, spec.metadata.version, result.spec):
        raise HTTPException(
            status_code=409,
            detail=[{
                "path": "metadata.version",
                "message": "Version conflict: another request updated this agent first. Re-fetch and retry.",
                "severity": "error",
            }],
        )
    return result


@app.post("/agents/{agent_id}/instructions", response_model=InstructionResponse)
def apply_agent_instruction(
    agent_id: str,
    body: InstructionRequest,
    store: AgentStore = Depends(get_store),
    client: Anthropic = Depends(get_anthropic_client),
) -> InstructionResponse:
    spec = _get_spec_or_404(store, agent_id)

    try:
        # Wraps the (test-overridable, via get_anthropic_client) Anthropic
        # client in the same AIProvider abstraction AI-node execution
        # uses — compile_instruction itself is now fully provider-agnostic
        # (backend/nl_compiler/compiler.py), this route just keeps
        # defaulting to Anthropic specifically, preserving this endpoint's
        # exact original behavior and test-override seam. The NEW
        # propose/apply chat routes below are where a request can actually
        # choose OpenAI instead.
        from backend.ai_providers.anthropic_provider import AnthropicProvider

        outcome = compile_instruction(spec, body.message, provider=AnthropicProvider(client=client))
    except NLCompileError as e:
        raise HTTPException(status_code=502, detail=str(e))

    if outcome.result.ok and outcome.result.spec is not None:
        # Same atomic guard as the mutations route — compile_instruction's
        # internal apply_mutations() already checked the version against
        # the `spec` snapshot fetched above, but only compare_and_set()
        # makes the check-and-write atomic against a concurrent request.
        if not store.compare_and_set(agent_id, spec.metadata.version, outcome.result.spec):
            raise HTTPException(
                status_code=409,
                detail={
                    "reasoning": outcome.reasoning,
                    "errors": [{
                        "path": "metadata.version",
                        "message": "Version conflict: another request updated this agent first. Re-fetch and retry.",
                        "severity": "error",
                    }],
                },
            )
    elif not outcome.result.ok:
        is_conflict = any(err.path == "metadata.version" for err in (outcome.result.errors or []))
        raise HTTPException(
            status_code=409 if is_conflict else 422,
            detail={
                "reasoning": outcome.reasoning,
                "errors": [err.model_dump() for err in (outcome.result.errors or [])],
            },
        )

    return InstructionResponse(reasoning=outcome.reasoning, attempts=outcome.attempts, result=outcome.result)


# ---- Conversational Workflow Builder: propose / preview / apply ---------
#
# This is the "AI proposes, a person decides, the mutation engine commits"
# flow the chat UI (packages/canvas/src/ChatPanel.tsx) uses — distinct
# from /instructions above (which auto-applies, preserved unchanged for
# backward compatibility). Both routes call the exact same
# compile_instruction() / apply_mutations() underneath; the only
# difference is WHEN (or whether) the result gets persisted.
#
# Reuses compile_instruction()'s existing purity: apply_mutations() never
# touches the AgentStore on its own, so "propose" is just "call
# compile_instruction and stop before the persistence step" — no new
# mutation/validation machinery, no proposal cache to keep in sync,
# nothing that could drift from what /instructions and the plain
# /mutations route already do. "Apply" re-runs apply_mutations() against
# a freshly-fetched spec (cheap and deterministic given the same
# mutations) rather than trusting a stale in-memory result, so a
# concurrent edit between propose and apply is still caught by the normal
# version-conflict path, not silently overwritten.


def _resolve_chat_provider(provider_name: str | None, model_name: str | None):
    """Resolves which AIProvider a chat request uses -- the server
    default (build_provider()) unless the request explicitly names one,
    in which case missing credentials are a clear 422, never a silent
    substitution -- same "explicit selection is never silently downgraded"
    rule already applied to per-node AI provider selection
    (compiler.py's _check_ai_provider_availability) and extended here to
    the conversational builder, per the same requirement."""
    from backend.ai_providers.registry import (
        ProviderConfigurationError,
        UnsupportedProviderError,
        api_key_env_var,
        build_provider,
        has_credentials,
        resolve_provider_name,
    )

    try:
        resolved_name = resolve_provider_name(provider_name)
    except UnsupportedProviderError as e:
        raise HTTPException(status_code=422, detail=str(e))

    if provider_name is not None and not has_credentials(resolved_name):
        env_var = api_key_env_var(resolved_name)
        raise HTTPException(
            status_code=503,
            detail={
                "error": (
                    f"This request explicitly selected provider '{resolved_name}', but {env_var} is not "
                    f"configured on this server. Set {env_var}, or omit 'provider' to use the server default."
                ),
                "reason": "no_api_key",
            },
        )
    if not has_credentials(resolved_name):
        raise HTTPException(
            status_code=503,
            detail={
                "error": (
                    f"No AI provider is configured on this server (checked '{resolved_name}'). "
                    f"Set {api_key_env_var(resolved_name)} to enable the chat builder."
                ),
                "reason": "no_api_key",
            },
        )
    try:
        return resolved_name, build_provider(resolved_name, model_name)
    except ProviderConfigurationError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/agents/{agent_id}/workflows/{workflow_id}/chat/propose", response_model=ChatProposalResponse)
def propose_chat_instruction(
    agent_id: str,
    workflow_id: str,
    body: ChatProposeRequest,
    store: AgentStore = Depends(get_store),
) -> ChatProposalResponse:
    spec = _get_workflow_or_404(store, agent_id, workflow_id)
    provider_name, provider = _resolve_chat_provider(body.provider, body.model)

    try:
        outcome = compile_instruction(spec, body.message, provider=provider)
    except NLCompileError as e:
        raise HTTPException(status_code=502, detail=str(e))

    return ChatProposalResponse(
        reasoning=outcome.reasoning,
        mutations=[m.model_dump(by_alias=True, mode="json") for m in outcome.mutations],
        ok=outcome.result.ok,
        errors=[err.model_dump() for err in (outcome.result.errors or [])],
        attempts=outcome.attempts,
        base_version=spec.metadata.version,
        provider=provider_name,
        model=body.model,
        preview_spec=outcome.result.spec.model_dump(by_alias=True, mode="json") if outcome.result.spec else None,
    )


@app.post("/agents/{agent_id}/workflows/{workflow_id}/chat/apply", response_model=AgentSpecification)
def apply_chat_proposal(
    agent_id: str,
    workflow_id: str,
    body: ChatApplyRequest,
    store: AgentStore = Depends(get_store),
) -> AgentSpecification:
    """Commits a previously-proposed mutation batch. Re-validates from
    scratch against the CURRENT stored spec (not whatever was true at
    propose time) — the same guarantee /mutations already gives every
    other kind of edit: "invalid AI mutations must be rejected without
    modifying the workflow" holds even if the workflow changed between
    propose and apply, because this is a real apply_mutations() call
    against live state, not a rubber-stamp of the preview."""
    spec = _get_workflow_or_404(store, agent_id, workflow_id)

    try:
        mutations = [parse_mutation(m) for m in body.mutations]
    except PydanticValidationError as e:
        raise HTTPException(status_code=422, detail=f"Malformed mutation: {e}")

    for mutation in mutations:
        if mutation.kind == "update_metadata" and "owner" in mutation.patch:
            raise HTTPException(status_code=422, detail="Cannot change an agent's owner via a chat-proposed mutation.")

    # force=True: rebase onto whatever is stored now (user confirmed overwrite).
    base_version = spec.metadata.version if body.force else body.expected_version
    result = apply_mutations(spec, mutations, expected_version=None if body.force else body.expected_version)
    if not result.ok or result.spec is None:
        # When not force and version mismatch, surface 409 so the UI can offer overwrite.
        msgs = [err.message for err in (result.errors or [])]
        if any("Version conflict" in m for m in msgs) and not body.force:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Version conflict: the workflow changed since this suggestion was made "
                    "(for example a node was moved). You can force-apply to overwrite on top of the current graph."
                ),
            )
        raise HTTPException(status_code=422, detail=[err.model_dump() for err in (result.errors or [])])

    cas_version = spec.metadata.version if body.force else body.expected_version
    if not store.compare_and_set(workflow_id, cas_version, result.spec):
        raise HTTPException(
            status_code=409,
            detail="Version conflict: this workflow changed again during apply. Retry or force-apply.",
        )
    return result.spec


@app.post("/agents/{agent_id}/compile", response_model=CompileCheckResponse)
def compile_check(agent_id: str, store: AgentStore = Depends(get_store)) -> CompileCheckResponse:
    spec = _get_spec_or_404(store, agent_id)
    try:
        compile_agent(spec)
    except CompileError as e:
        return CompileCheckResponse(ok=False, error=str(e), node_id=e.node_id)
    return CompileCheckResponse(ok=True)




def _scheduler_run_agent(agent_id: str) -> None:
    """Invoke a scheduled agent run using the same path as POST /run."""
    store = get_store()
    overrides = get_executor_overrides()
    spec = store.get(agent_id)
    overrides = _resolve_executor_overrides(spec, overrides)
    graph = compile_agent(spec, checkpointer=store.checkpointer(agent_id), executor_overrides=overrides)
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    graph.invoke({"variables": {}, "trace": []}, config=config)


@app.on_event("startup")
def _start_background_scheduler() -> None:
    import os
    if os.environ.get("DISABLE_SCHEDULER", "").strip() in ("1", "true", "yes"):
        return
    from backend.scheduler import start_scheduler

    def store_factory():
        return get_store()

    start_scheduler(store_factory, _scheduler_run_agent)


@app.on_event("shutdown")
def _stop_background_scheduler() -> None:
    from backend.scheduler import stop_scheduler
    stop_scheduler()


@app.get("/scheduler/status")
def scheduler_status() -> dict:
    from backend.scheduler import _last_run, _thread
    return {
        "running": bool(_thread and _thread.is_alive()),
        "last_runs": dict(_last_run),
    }

@app.post("/agents/{agent_id}/run", response_model=RunResponse)
def run_agent(
    agent_id: str,
    body: RunRequest,
    store: AgentStore = Depends(get_store),
    executor_overrides: dict = Depends(get_executor_overrides),
) -> RunResponse:
    spec = _get_spec_or_404(store, agent_id)
    executor_overrides = _resolve_executor_overrides(spec, executor_overrides)

    try:
        graph = compile_agent(spec, checkpointer=store.checkpointer(agent_id), executor_overrides=executor_overrides)
    except CompileError as e:
        raise HTTPException(
            status_code=422,
            detail={"error": f"Spec does not compile: {e}", "node_id": e.node_id},
        )

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    try:
        result = graph.invoke({"variables": body.seed_variables, "trace": []}, config=config)
    except RuntimeError as e:
        return _partial_result_after_failure(graph, config, thread_id, e)

    return _to_run_response(thread_id, result)


@app.post("/agents/{agent_id}/resume", response_model=RunResponse)
def resume_agent(
    agent_id: str,
    body: ResumeRequest,
    store: AgentStore = Depends(get_store),
    executor_overrides: dict = Depends(get_executor_overrides),
) -> RunResponse:
    spec = _get_spec_or_404(store, agent_id)
    executor_overrides = _resolve_executor_overrides(spec, executor_overrides)

    try:
        graph = compile_agent(spec, checkpointer=store.checkpointer(agent_id), executor_overrides=executor_overrides)
    except CompileError as e:
        raise HTTPException(
            status_code=422,
            detail={"error": f"Spec does not compile: {e}", "node_id": e.node_id},
        )

    config = {"configurable": {"thread_id": body.thread_id}}
    try:
        result = graph.invoke(Command(resume=body.resume), config=config)
    except RuntimeError as e:
        return _partial_result_after_failure(graph, config, body.thread_id, e)

    return _to_run_response(body.thread_id, result)


def _partial_result_after_failure(graph, config: dict, thread_id: str, error: Exception) -> RunResponse:
    """
    Called from every `graph.invoke()`/`graph.stream()` failure handler
    (`/run`, `/resume`, and `_stream_run`'s final `done` event on
    failure). Before this function existed, all three just returned
    `trace=[], variables={}` on any RuntimeError — discarding every
    successful node's output the moment a LATER node failed, even though
    LangGraph's checkpointer (compile_agent is always given one — see
    store.checkpointer()) already persisted that state after each
    successful node. A run that classified an invoice, extracted its
    fields, saved them to the database, and only THEN failed sending the
    email would report back as if NOTHING had happened.

    `graph.get_state(config)` reads that already-persisted state back —
    same mechanism `_stream_run` already relied on for resume (see its
    `existing_state` usage above) — and `state.next` (the node(s) that
    were about to run when execution stopped) is genuinely the failed
    node's id, not a guess: LangGraph only advances `next` past a node
    after that node's write succeeds, so if a node raised, it's still
    sitting in `next` when we read the state back.
    """
    state = graph.get_state(config)
    values = state.values or {}
    failed_node_id = state.next[0] if state.next else None
    return RunResponse(
        thread_id=thread_id,
        status="failed",
        trace=values.get("trace", []),
        variables=values.get("variables", {}),
        error=str(error),
        failed_node_id=failed_node_id,
    )


def _to_run_response(thread_id: str, result: dict) -> RunResponse:
    interrupts = result.get("__interrupt__")
    if interrupts:
        return RunResponse(
            thread_id=thread_id,
            status="paused",
            trace=result.get("trace", []),
            variables=result.get("variables", {}),
            interrupt=interrupts[0].value,
        )
    return RunResponse(
        thread_id=thread_id,
        status="completed",
        trace=result.get("trace", []),
        variables=result.get("variables", {}),
    )


@app.post("/agents/{agent_id}/run/stream")
def run_agent_stream(
    agent_id: str,
    body: RunRequest,
    store: AgentStore = Depends(get_store),
    executor_overrides: dict = Depends(get_executor_overrides),
):
    """
    Server-Sent Events version of /run: emits one `trace` event per node
    as it actually executes, then a final `done` event carrying the same
    shape /run returns synchronously. This is what a live-updating trace
    panel (doc section 12) subscribes to — the UI can highlight a node the
    moment it fires instead of waiting for the whole run (or the whole
    pause) to resolve.
    """
    spec = _get_spec_or_404(store, agent_id)
    executor_overrides = _resolve_executor_overrides(spec, executor_overrides)
    try:
        graph = compile_agent(spec, checkpointer=store.checkpointer(agent_id), executor_overrides=executor_overrides)
    except CompileError as e:
        raise HTTPException(
            status_code=422,
            detail={"error": f"Spec does not compile: {e}", "node_id": e.node_id},
        )

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    return StreamingResponse(
        _stream_run(graph, {"variables": body.seed_variables, "trace": []}, config, thread_id),
        media_type="text/event-stream",
    )


@app.post("/agents/{agent_id}/resume/stream")
def resume_agent_stream(
    agent_id: str,
    body: ResumeRequest,
    store: AgentStore = Depends(get_store),
    executor_overrides: dict = Depends(get_executor_overrides),
):
    spec = _get_spec_or_404(store, agent_id)
    executor_overrides = _resolve_executor_overrides(spec, executor_overrides)
    try:
        graph = compile_agent(spec, checkpointer=store.checkpointer(agent_id), executor_overrides=executor_overrides)
    except CompileError as e:
        raise HTTPException(
            status_code=422,
            detail={"error": f"Spec does not compile: {e}", "node_id": e.node_id},
        )

    config = {"configurable": {"thread_id": body.thread_id}}
    return StreamingResponse(
        _stream_run(graph, Command(resume=body.resume), config, body.thread_id),
        media_type="text/event-stream",
    )


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _stream_run(graph, graph_input, config: dict, thread_id: str):
    """
    Shared generator behind both streaming routes: walks
    graph.stream(..., stream_mode=["values", "custom"]) — "values" gives
    the growing trace/variables after each node completes (same as before),
    "custom" delivers the node_started signal each node's writer() call
    emits (see compiler.py's _make_node_fn) genuinely live, i.e. BEFORE
    that node's own completion shows up in the "values" stream. Confirmed
    this interleaving is real (not buffered) — see the timing proof this
    file's test (test_streaming.py) and the README's Execution Trace UI
    section reference.

    Emits one SSE `node_started` event right as each node begins, one SSE
    `trace` event per newly-appended trace entry as nodes finish, then a
    final `done` event with the same RunResponse shape /run and /resume
    return synchronously.
    """
    # For a resumed run, the checkpointed state ALREADY contains the trace
    # from before the pause — graph.stream() re-yields the full
    # accumulated trace in its first "values" chunk, not just what's new.
    # Seeding seen_trace_len from the persisted state (not 0) avoids
    # re-emitting every prior node's trace event a second time on resume.
    # get_state() on a thread that's never been run returns empty values,
    # so this is correct for fresh runs too (seen_trace_len starts at 0).
    existing_state = graph.get_state(config)
    seen_trace_len = len((existing_state.values or {}).get("trace", []))
    last_values_chunk: dict = {}
    try:
        for mode, chunk in graph.stream(graph_input, config=config, stream_mode=["values", "custom"]):
            if mode == "custom":
                # _make_node_fn's writer() emits two distinct custom event
                # shapes: {"type": "node_started", ...} when a node
                # begins, and {"type": "node_failed", ...} if it raises.
                # Forwarding chunk["type"] as the SSE event name (instead
                # of hardcoding "node_started" for both) means a listening
                # client actually gets told a node failed as it happens,
                # not just via the final `done` event after the fact.
                yield _sse(chunk.get("type", "node_started"), chunk)
            elif mode == "values":
                last_values_chunk = chunk
                trace = chunk.get("trace", [])
                for event in trace[seen_trace_len:]:
                    yield _sse("trace", event)
                seen_trace_len = len(trace)
    except RuntimeError as e:
        yield _sse("done", _partial_result_after_failure(graph, config, thread_id, e).model_dump())
        return

    run_response = _to_run_response(thread_id, last_values_chunk)
    yield _sse("done", run_response.model_dump())


@app.post("/agents/{agent_id}/publish", response_model=PublishResponse)
def publish_agent(agent_id: str, store: AgentStore = Depends(get_store)) -> PublishResponse:
    spec = _get_spec_or_404(store, agent_id)
    result = publish_spec(spec, expected_version=spec.metadata.version)
    if not result.ok:
        raise HTTPException(status_code=422, detail=[err.model_dump() for err in (result.errors or [])])
    if not store.compare_and_set(agent_id, spec.metadata.version, result.spec):
        raise HTTPException(
            status_code=409,
            detail=[{
                "path": "metadata.version",
                "message": "Version conflict: another request updated this agent first. Re-fetch and retry.",
                "severity": "error",
            }],
        )
    return PublishResponse(spec=result.spec, published_version=result.spec.metadata.published_version)
