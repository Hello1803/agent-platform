"""
The definitive persistence proof: starts the REAL uvicorn server as a
subprocess with DATABASE_URL set, creates an agent and pauses a run on it
over genuine HTTP, then KILLS THAT PROCESS OUTRIGHT and starts a brand-new
uvicorn subprocess with the same DATABASE_URL — a literal process restart,
not a dependency-override simulation of one — and confirms the second
process can resume the run the first process never got to finish.

This is what backend/persistence/test_postgres_persistence.py already
proved at the Python-object level (two independent PostgresAgentStore
instances); this test proves the same thing one layer up, at the level
that actually matters in production: the HTTP server process itself.

Requires a running Postgres reachable at DATABASE_URL (see README's
Postgres setup section) and requests installed.

Run:
    PYTHONPATH=. python backend/persistence/test_server_restart.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
PORT = 8942
BASE_URL = f"http://127.0.0.1:{PORT}"
DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://agent_platform:agent_platform_dev@localhost/agent_platform"
)

passed = 0
failed = 0


def check(condition: bool, message: str) -> None:
    global passed, failed
    if condition:
        passed += 1
    else:
        failed += 1
        print(f"FAIL: {message}")


def start_server(log_path: Path) -> subprocess.Popen:
    env = {**os.environ, "PYTHONPATH": str(ROOT), "DATABASE_URL": DATABASE_URL}
    log_file = open(log_path, "w")
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.api.app:app", "--port", str(PORT), "--host", "127.0.0.1"],
        cwd=str(ROOT), env=env, stdout=log_file, stderr=subprocess.STDOUT,
    )


def wait_for_server(timeout: float = 20.0) -> None:
    deadline = time.time() + timeout
    last_error = None
    while time.time() < deadline:
        try:
            httpx.get(f"{BASE_URL}/agents/__probe__", timeout=5.0)
            return
        except (httpx.ConnectError, httpx.ReadTimeout) as e:
            last_error = e
            time.sleep(0.3)
    raise RuntimeError(f"Server did not become ready in time: {last_error}")


def load_spec_json(name: str) -> dict:
    return json.loads((ROOT / "examples" / name).read_text())


def main() -> None:
    agent_id = f"agt_restart_{uuid.uuid4().hex[:8]}"

    print("=== Process #1: starting uvicorn with DATABASE_URL set ===")
    proc_1 = start_server(ROOT / "backend" / "persistence" / "_server1.log")
    try:
        wait_for_server()
        print(f"  server up on {BASE_URL}, pid={proc_1.pid}")

        spec = load_spec_json("invoice-agent.json")
        spec["metadata"]["id"] = agent_id
        resp = httpx.post(f"{BASE_URL}/agents", json={"spec": spec}, timeout=10.0)
        check(resp.status_code == 201, f"create should 201: {resp.text}")
        print(f"  created {agent_id}")

        resp = httpx.post(f"{BASE_URL}/agents/{agent_id}/run", json={"seed_variables": {}}, timeout=10.0)
        check(resp.status_code == 200, f"run should 200: {resp.text}")
        run_body = resp.json()
        check(run_body["status"] == "paused", f"expected paused, got {run_body['status']}")
        thread_id = run_body["thread_id"]
        print(f"  run paused at human_approval, thread_id={thread_id}")

    finally:
        print(f"  killing process #1 (pid={proc_1.pid}) outright — SIGKILL, no graceful shutdown")
        proc_1.kill()
        proc_1.wait(timeout=10)

    print("\n=== Process #2: a genuinely new uvicorn process, same DATABASE_URL ===")
    proc_2 = start_server(ROOT / "backend" / "persistence" / "_server2.log")
    try:
        wait_for_server()
        print(f"  server up on {BASE_URL}, pid={proc_2.pid} (different pid than before: {proc_2.pid != proc_1.pid})")
        check(proc_2.pid != proc_1.pid, "the two servers are genuinely different OS processes")

        resp = httpx.get(f"{BASE_URL}/agents/{agent_id}", timeout=10.0)
        check(resp.status_code == 200, f"get should 200 from the new process: {resp.text}")
        print("  OK: spec correctly readable from a process that never created it")

        resp = httpx.post(
            f"{BASE_URL}/agents/{agent_id}/resume",
            json={"thread_id": thread_id, "resume": {"approved": True}},
            timeout=10.0,
        )
        check(resp.status_code == 200, f"resume should 200 from the new process: {resp.text}")
        resumed = resp.json()
        check(resumed["status"] == "completed", f"expected completed, got {resumed['status']}")
        check(resumed["variables"].get("email_sent") is True, "run completed correctly after a literal process restart")
        print("  OK: a paused run, started by a process that no longer exists, was resumed to completion by a new one")

    finally:
        proc_2.kill()
        proc_2.wait(timeout=10)

    print(f"\n{passed} passed, {failed} failed")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
