"""
Regression tests for backend/real_executors/tools.py — the fix for the
"silent stub fallback" gap in Runtime Executors (P0.2): a mode="connected"
tool binding must genuinely attempt the real call and raise on failure,
never quietly return the same canned data a mode="demo" binding would.

Three kinds of proof, deliberately different for different reasons:

  1. demo mode: proven by calling the executor directly and checking it
     produces byte-for-byte the same output as the plain stub in
     compiler/executors.py, regardless of whether a credential exists.
  2. connected mode against a genuinely reachable, allowlisted domain
     (api_call against pypi.org, read_file/write_file against a real
     scoped local directory): proven end-to-end, no mocking.
  3. connected mode against Gmail/Sheets (gmail.googleapis.com,
     sheets.googleapis.com — NOT in this sandbox's egress allowlist, see
     network_configuration): proven with an injected httpx.MockTransport
     that asserts the request was built correctly (method, URL, auth
     header, body) and returns a realistic response — the same
     "inject a fake client instead of needing live access" pattern the
     NL compiler's own tests use for Anthropic (FakeAnthropicClient in
     nl_compiler/test_compiler.py).

Run:
    PYTHONPATH=. python backend/real_executors/test_tools.py
"""
from __future__ import annotations

import base64
import json
import tempfile
from pathlib import Path

import httpx

from backend.compiler.compiler import CompileError, compile_agent
from backend.compiler.executors import ToolPermissionError
from backend.credentials.store import CredentialStore
from backend.models.agent_spec import AgentNode, AgentSpecification, ToolBinding
from backend.real_executors.tools import (
    ConnectedExecutionError,
    build_api_call_executor,
    build_connected_tool_executors,
    build_read_email_executor,
    build_read_file_executor,
    build_send_email_executor,
    build_write_file_executor,
)
from cryptography.fernet import Fernet


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _minimal_spec(nodes: list[dict], tools: list[dict], owner: str = "alice") -> AgentSpecification:
    return AgentSpecification.model_validate(
        {
            "metadata": {
                "id": "agt_test",
                "name": "Test",
                "status": "draft",
                "version": "0.0.1",
                "createdAt": "2026-01-01T00:00:00Z",
                "updatedAt": "2026-01-01T00:00:00Z",
                "owner": owner,
            },
            "goal": "Test fixture agent for real_executors/tools.py",
            "nodes": [
                {"id": "n_trigger", "type": "trigger_manual", "category": "trigger", "label": "Start"},
                *nodes,
            ],
            "edges": [{"id": f"e_{n['id']}", "source": "n_trigger", "target": n["id"]} for n in nodes],
            "tools": tools,
        }
    )


def main() -> None:
    store = CredentialStore(fernet=Fernet(Fernet.generate_key()))

    # =====================================================================
    print("=== demo mode: identical to the plain stub, credential or not ===")
    spec = _minimal_spec(
        nodes=[{"id": "n_send", "type": "send_email", "category": "tool", "label": "Send", "toolRefs": ["t1"]}],
        tools=[{"id": "t1", "toolId": "gmail.send", "permissions": ["send"], "mode": "demo"}],
    )
    node = next(n for n in spec.nodes if n.id == "n_send")
    executor = build_send_email_executor(credential_store=store)
    result = executor(node, spec, {})
    check(result == {"email_sent": True}, f"demo mode should match the plain stub exactly, got {result}")
    print("  OK: demo-mode send_email matches the plain stub's output exactly")

    # =====================================================================
    print("\n=== connected mode, no credential configured: fails loudly, not silently ===")
    spec2 = _minimal_spec(
        nodes=[{"id": "n_send", "type": "send_email", "category": "tool", "label": "Send", "toolRefs": ["t1"]}],
        tools=[{"id": "t1", "toolId": "gmail.send", "permissions": ["send"], "mode": "connected", "credentialRef": "cred_ghost"}],
    )
    # Bypass compile-time validation on purpose (this simulates a
    # credential that existed at compile time and was deleted before run —
    # the exact "hot-swap between compile and run" scenario the executor
    # docstrings call out) to prove the RUNTIME check independently
    # catches it too, not just the compiler.
    node2 = next(n for n in spec2.nodes if n.id == "n_send")
    executor2 = build_send_email_executor(credential_store=store)
    try:
        executor2(node2, spec2, {})
        raise AssertionError("connected send_email with a nonexistent credential should raise, not return success")
    except Exception as exc:  # CredentialNotFoundError from the store
        check("cred_ghost" in str(exc) or type(exc).__name__ == "CredentialNotFoundError", f"unexpected error: {exc!r}")
        print(f"  OK: raised {type(exc).__name__} instead of a fake success result")

    # =====================================================================
    print("\n=== connected mode: Gmail send, request built correctly (mocked transport) ===")
    gmail_cred = store.create(
        provider="gmail",
        name="test",
        secret={"access_token": "tok_abc123", "refresh_token": "r", "client_id": "c", "client_secret": "s"},
        owner_id="alice",
    )
    captured: dict = {}

    def gmail_handler(request: httpx.Request) -> httpx.Response:
        captured["method"] = request.method
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"id": "msg_real_id_123"})

    transport = httpx.MockTransport(gmail_handler)
    spec3 = _minimal_spec(
        nodes=[
            {
                "id": "n_send",
                "type": "send_email",
                "category": "tool",
                "label": "Send",
                "toolRefs": ["t1"],
                "config": {"to": "vendor@example.com", "subject": "Invoice received", "body_variable": "draft_email"},
            }
        ],
        tools=[{"id": "t1", "toolId": "gmail.send", "permissions": ["send"], "mode": "connected", "credentialRef": gmail_cred.id}],
    )
    node3 = next(n for n in spec3.nodes if n.id == "n_send")
    executor3 = build_send_email_executor(credential_store=store, transport=transport)
    result3 = executor3(node3, spec3, {"draft_email": "Thanks, we received your invoice."})
    check(result3 == {"email_sent": True, "email_message_id": "msg_real_id_123"}, f"unexpected result: {result3}")
    check(captured["method"] == "POST", "should POST to send a message")
    check(captured["url"] == "https://gmail.googleapis.com/gmail/v1/users/me/messages/send", f"wrong URL: {captured['url']}")
    check(captured["auth"] == "Bearer tok_abc123", f"wrong auth header: {captured['auth']}")
    raw = base64.urlsafe_b64decode(captured["body"]["raw"]).decode()
    check("vendor@example.com" in raw and "Thanks, we received your invoice." in raw, "MIME message should contain the real recipient and body")
    print("  OK: real Gmail send request built with correct method/URL/auth/body")

    # =====================================================================
    print("\n=== connected mode: Gmail send, live call fails -> raises, not fake success ===")

    def failing_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "invalid_token"})

    failing_transport = httpx.MockTransport(failing_handler)
    executor_fail = build_send_email_executor(credential_store=store, transport=failing_transport)
    try:
        executor_fail(node3, spec3, {"draft_email": "x"})
        raise AssertionError("a 401 from Gmail should raise, not return email_sent: True")
    except ConnectedExecutionError as exc:
        check("401" in str(exc), f"error should mention the status code: {exc}")
        print(f"  OK: a real 401 from the provider raises ConnectedExecutionError, not a fake success")

    # =====================================================================
    print("\n=== connected mode: Gmail read, request + response parsing ===")

    def read_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/messages"):
            return httpx.Response(200, json={"messages": [{"id": "m1"}]})
        return httpx.Response(
            200,
            json={
                "snippet": "Please find attached...",
                "payload": {"headers": [{"name": "Subject", "value": "Invoice #402"}, {"name": "From", "value": "billing@acme.com"}]},
            },
        )

    read_executor = build_read_email_executor(credential_store=store, transport=httpx.MockTransport(read_handler))
    spec4 = _minimal_spec(
        nodes=[{"id": "n_read", "type": "read_email", "category": "tool", "label": "Read", "toolRefs": ["t1"]}],
        tools=[{"id": "t1", "toolId": "gmail.read", "permissions": ["read"], "mode": "connected", "credentialRef": gmail_cred.id}],
    )
    node4 = next(n for n in spec4.nodes if n.id == "n_read")
    result4 = read_executor(node4, spec4, {})
    check(result4["email"]["subject"] == "Invoice #402", f"unexpected result: {result4}")
    check(result4["email"]["from"] == "billing@acme.com", f"unexpected result: {result4}")
    print("  OK: real Gmail read parses subject/from/snippet out of the actual API response shape")

    # =====================================================================
    print("\n=== connected mode: api_call against a REAL, allowlisted domain (pypi.org) ===")
    api_executor = build_api_call_executor(credential_store=store)
    spec5 = _minimal_spec(
        nodes=[
            {
                "id": "n_api",
                "type": "api_call",
                "category": "tool",
                "label": "Check package",
                "toolRefs": ["t1"],
                "config": {"url": "https://pypi.org/pypi/requests/json", "method": "GET"},
            }
        ],
        tools=[{"id": "t1", "toolId": "http.request", "permissions": ["read"], "mode": "connected"}],
    )
    node5 = next(n for n in spec5.nodes if n.id == "n_api")
    result5 = api_executor(node5, spec5, {})
    check(result5["api_status"] == 200, f"expected a real 200 from pypi.org, got {result5}")
    check(result5["api_response"]["info"]["name"].lower() == "requests", f"unexpected payload: {result5}")
    print("  OK: connected api_call made a genuinely live HTTP request to an allowlisted domain and parsed the real response")

    print("\n=== api_call with no tool binding at all: falls back to the demo/generic stub, doesn't error ===")
    spec6 = _minimal_spec(
        nodes=[{"id": "n_api2", "type": "api_call", "category": "tool", "label": "Notify", "toolRefs": []}],
        tools=[],
    )
    node6 = spec6.nodes[-1]
    # tool_refs=[] on a node with no REQUIRED_CAPABILITY entry is valid per
    # the compiler (api_call isn't in REQUIRED_CAPABILITY) — this is
    # exactly the shape the NL compiler produces for a lightweight
    # notification node with no binding yet.
    result6 = api_executor(node6, spec6, {})
    check("n_api2_result" in result6, f"no-binding api_call should fall back to the demo stub, got {result6}")
    print("  OK: an api_call node with no tool binding still runs (demo/generic stub), never mistaken for 'connected'")

    # =====================================================================
    print("\n=== connected mode: read_file genuinely parses a real CSV (extraction, not a stub) ===")
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "invoices.csv"
        csv_path.write_text(
            "invoice_id,vendor,amount,status\n"
            "INV-1,Acme Corp,14250.00,unpaid\n"
            "INV-2,Globex Inc,8320.50,unpaid\n"
            "INV-3,Acme Corp,5600.00,paid\n"
        )
        csv_read_executor = build_read_file_executor(base_dir=tmp)
        spec_csv = _minimal_spec(
            nodes=[
                {
                    "id": "n_read_csv",
                    "type": "read_file",
                    "category": "tool",
                    "label": "Read invoices",
                    "toolRefs": ["t1"],
                    "config": {"path": "invoices.csv"},
                }
            ],
            tools=[{"id": "t1", "toolId": "files.read", "permissions": ["read"], "mode": "connected"}],
        )
        node_csv = next(n for n in spec_csv.nodes if n.id == "n_read_csv")
        result_csv = csv_read_executor(node_csv, spec_csv, {})

        check(result_csv["table_columns"] == ["invoice_id", "vendor", "amount", "status"], f"unexpected columns: {result_csv['table_columns']}")
        check(result_csv["table_row_count"] == 3, f"expected 3 rows, got {result_csv['table_row_count']}")
        check(len(result_csv["rows"]) == 3, f"expected 3 full rows, got {len(result_csv['rows'])}")
        check(result_csv["rows"][0]["vendor"] == "Acme Corp", f"unexpected first row: {result_csv['rows'][0]}")
        # The real point: these are computed from the actual file contents,
        # not canned. 14250.00 + 8320.50 + 5600.00 = 28170.50, genuinely summed.
        amount_summary = result_csv["table_summary"]["amount"]
        check(abs(amount_summary["sum"] - 28170.50) < 0.001, f"sum should be genuinely computed, got {amount_summary}")
        check(abs(amount_summary["max"] - 14250.00) < 0.001, f"max should be genuinely computed, got {amount_summary}")
        check(abs(amount_summary["min"] - 5600.00) < 0.001, f"min should be genuinely computed, got {amount_summary}")
        check(amount_summary["count"] == 3, f"count should be genuinely computed, got {amount_summary}")
        print(f"  OK: real column extraction — {result_csv['table_columns']}")
        print(f"  OK: real row count — {result_csv['table_row_count']}")
        print(f"  OK: real computed sum/min/max on 'amount' — {amount_summary}")

        print("\n=== connected mode: read_file genuinely parses a real XLSX ===")
        xlsx_path = Path(tmp) / "invoices.xlsx"
        import pandas as pd

        pd.DataFrame(
            {
                "invoice_id": ["INV-10", "INV-11"],
                "vendor": ["Initech", "Umbrella LLC"],
                "amount": [2199.99, 19999.00],
            }
        ).to_excel(xlsx_path, index=False)
        xlsx_read_executor = build_read_file_executor(base_dir=tmp)
        spec_xlsx = _minimal_spec(
            nodes=[
                {
                    "id": "n_read_xlsx",
                    "type": "read_file",
                    "category": "tool",
                    "label": "Read invoices",
                    "toolRefs": ["t1"],
                    "config": {"path": "invoices.xlsx"},
                }
            ],
            tools=[{"id": "t1", "toolId": "files.read", "permissions": ["read"], "mode": "connected"}],
        )
        node_xlsx = next(n for n in spec_xlsx.nodes if n.id == "n_read_xlsx")
        result_xlsx = xlsx_read_executor(node_xlsx, spec_xlsx, {})
        check(result_xlsx["table_row_count"] == 2, f"unexpected: {result_xlsx}")
        check(abs(result_xlsx["table_summary"]["amount"]["sum"] - 22198.99) < 0.001, f"unexpected: {result_xlsx}")
        print(f"  OK: real .xlsx extraction — {result_xlsx['table_columns']}, sum={result_xlsx['table_summary']['amount']['sum']}")

        print("\n=== connected mode: a malformed table raises, doesn't silently return an empty result ===")
        bad_path = Path(tmp) / "broken.xlsx"
        bad_path.write_bytes(b"this is not a real spreadsheet file")
        bad_spec = _minimal_spec(
            nodes=[
                {
                    "id": "n_read_bad",
                    "type": "read_file",
                    "category": "tool",
                    "label": "Read",
                    "toolRefs": ["t1"],
                    "config": {"path": "broken.xlsx"},
                }
            ],
            tools=[{"id": "t1", "toolId": "files.read", "permissions": ["read"], "mode": "connected"}],
        )
        node_bad = next(n for n in bad_spec.nodes if n.id == "n_read_bad")
        try:
            build_read_file_executor(base_dir=tmp)(node_bad, bad_spec, {})
            raise AssertionError("a corrupt xlsx file should raise, not silently return an empty table")
        except ConnectedExecutionError as exc:
            check("Could not parse" in str(exc), f"unexpected error: {exc}")
            print("  OK: a corrupt spreadsheet raises ConnectedExecutionError instead of a fake/empty success")

    # =====================================================================
    print("\n=== connected mode: read_file / write_file against a REAL scoped local directory ===")
    with tempfile.TemporaryDirectory() as tmp:
        write_executor = build_write_file_executor(base_dir=tmp)
        read_executor2 = build_read_file_executor(base_dir=tmp)

        spec7 = _minimal_spec(
            nodes=[
                {
                    "id": "n_write",
                    "type": "write_file",
                    "category": "tool",
                    "label": "Write",
                    "toolRefs": ["t1"],
                    "config": {"path": "notes/summary.txt", "content_variable": "summary"},
                }
            ],
            tools=[{"id": "t1", "toolId": "files.write", "permissions": ["write"], "mode": "connected"}],
        )
        node7 = next(n for n in spec7.nodes if n.id == "n_write")
        write_result = write_executor(node7, spec7, {"summary": "Q3 revenue is up 12%."})
        check(write_result["file_written"] is True, f"unexpected: {write_result}")
        on_disk = (Path(tmp) / "notes" / "summary.txt").read_text()
        check(on_disk == "Q3 revenue is up 12%.", f"file on disk doesn't match what was written: {on_disk!r}")
        print("  OK: connected write_file genuinely wrote a real file to disk")

        spec8 = _minimal_spec(
            nodes=[
                {
                    "id": "n_read2",
                    "type": "read_file",
                    "category": "tool",
                    "label": "Read",
                    "toolRefs": ["t1"],
                    "config": {"path": "notes/summary.txt"},
                }
            ],
            tools=[{"id": "t1", "toolId": "files.read", "permissions": ["read"], "mode": "connected"}],
        )
        node8 = next(n for n in spec8.nodes if n.id == "n_read2")
        read_result = read_executor2(node8, spec8, {})
        check(read_result["file_contents"] == "Q3 revenue is up 12%.", f"unexpected: {read_result}")
        print("  OK: connected read_file genuinely read the real file back")

        print("\n=== path traversal is refused, not silently resolved ===")
        spec9 = _minimal_spec(
            nodes=[
                {
                    "id": "n_read3",
                    "type": "read_file",
                    "category": "tool",
                    "label": "Read",
                    "toolRefs": ["t1"],
                    "config": {"path": "../../etc/passwd"},
                }
            ],
            tools=[{"id": "t1", "toolId": "files.read", "permissions": ["read"], "mode": "connected"}],
        )
        node9 = next(n for n in spec9.nodes if n.id == "n_read3")
        try:
            read_executor2(node9, spec9, {})
            raise AssertionError("a path escaping the scoped directory should be refused")
        except ConnectedExecutionError as exc:
            check("escapes" in str(exc), f"unexpected error: {exc}")
            print("  OK: a path escaping the agent's scoped file directory is refused")

    # =====================================================================
    print("\n=== end-to-end: compile_agent + real graph invocation with connected api_call ===")
    overrides = build_connected_tool_executors(credential_store=store)
    spec10 = _minimal_spec(
        nodes=[
            {
                "id": "n_api3",
                "type": "api_call",
                "category": "tool",
                "label": "Check package",
                "toolRefs": ["t1"],
                "config": {"url": "https://pypi.org/pypi/requests/json", "method": "GET"},
            }
        ],
        tools=[{"id": "t1", "toolId": "http.request", "permissions": ["read"], "mode": "connected"}],
    )
    graph = compile_agent(spec10, executor_overrides=overrides, credential_store=store)
    final_state = graph.invoke({"variables": {}, "trace": []}, config={"configurable": {"thread_id": "t1"}})
    check(final_state["variables"]["api_status"] == 200, f"unexpected final state: {final_state['variables']}")
    print("  OK: a full compiled graph run genuinely hit a live allowlisted endpoint through the real executor")

    print("\nAll real-executor mode-awareness checks passed.")


if __name__ == "__main__":
    main()
