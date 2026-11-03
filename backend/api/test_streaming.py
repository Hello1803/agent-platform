"""
Tests the SSE streaming endpoints (/run/stream, /resume/stream) against a
real running uvicorn server over genuine HTTP — the point of streaming is
that trace events arrive incrementally as each node actually executes,
not all at once when the run finishes, so this test specifically checks
that property (via wall-clock ordering of received events), not just that
the final result is correct.

Run:
    PYTHONPATH=. python backend/api/test_streaming.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
PORT = 8953
BASE_URL = f"http://127.0.0.1:{PORT}"

passed = 0
failed = 0


def check(condition: bool, message: str) -> None:
    global passed, failed
    if condition:
        passed += 1
    else:
        failed += 1
        print(f"FAIL: {message}")


def start_server() -> subprocess.Popen:
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    env.pop("DATABASE_URL", None)  # in-memory store is fine for this test
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.api.app:app", "--port", str(PORT), "--host", "127.0.0.1"],
        cwd=str(ROOT), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def wait_for_server(timeout: float = 15.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            httpx.get(f"{BASE_URL}/agents/__probe__", timeout=2.0)
            return
        except httpx.ConnectError:
            time.sleep(0.3)
    raise RuntimeError("Server did not become ready in time")


def parse_sse_stream(response: httpx.Response):
    """Yields (event_name, data_dict, wall_clock_time) for each SSE message."""
    event_name = None
    for line in response.iter_lines():
        if line.startswith("event:"):
            event_name = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            data = json.loads(line.split(":", 1)[1].strip())
            yield event_name, data, time.monotonic()


def main() -> None:
    print("Starting server...")
    server = start_server()
    try:
        wait_for_server()
        print("Server up.\n")

        spec = json.loads((ROOT / "examples" / "invoice-agent.json").read_text())
        spec["metadata"]["id"] = "agt_stream_test"
        resp = httpx.post(f"{BASE_URL}/agents", json={"spec": spec}, timeout=10.0)
        check(resp.status_code == 201, f"create should 201: {resp.text}")

        print("=== Streaming /run ===")
        received = []
        with httpx.stream(
            "POST", f"{BASE_URL}/agents/agt_stream_test/run/stream", json={"seed_variables": {}}, timeout=20.0
        ) as response:
            check(response.status_code == 200, f"stream should 200: {response.status_code}")
            for name, data, t in parse_sse_stream(response):
                received.append((name, data, t))
                if name == "node_started":
                    print(f"  [started] {data['node_label']}")
                elif name == "trace":
                    print(f"  [trace] {data['node_label']}: {data['detail']}")
                elif name == "done":
                    print(f"  [done] status={data['status']}")

        started_events = [e for e in received if e[0] == "node_started"]
        trace_events = [e for e in received if e[0] == "trace"]
        done_events = [e for e in received if e[0] == "done"]

        check(
            len(started_events) == len(trace_events) + 1,
            f"expected one node_started per trace event PLUS one for the paused human_approval node "
            f"(which starts but never completes to a trace event) = {len(trace_events) + 1}, got {len(started_events)}",
        )
        check(
            started_events[-1][1]["node_label"] == "Human Approval",
            "the extra (unmatched) node_started is for the node that actually paused — a genuinely useful "
            "signal ('now waiting for you') that trace alone, which only records completions, wouldn't show",
        )
        check(len(trace_events) >= 5, f"expected at least 5 trace events before pausing, got {len(trace_events)}")
        check(len(done_events) == 1, f"expected exactly one done event, got {len(done_events)}")
        check(done_events[0][1]["status"] == "paused", f"expected paused, got {done_events[0][1]['status']}")
        check(
            done_events[0][1]["interrupt"]["type"] == "human_approval",
            "done event correctly carries the interrupt payload",
        )

        # The property that actually matters for node_started: each node's
        # "started" event must arrive at or before that SAME node's "trace"
        # (completion) event — proving the started signal is genuinely a
        # pre-execution signal, not just a relabeled completion event.
        for i, matching_trace in enumerate(trace_events):
            data = started_events[i][1]
            t = started_events[i][2]
            check(
                data["node_id"] == matching_trace[1]["node_id"],
                f"node_started #{i} ({data['node_id']}) pairs with trace #{i} ({matching_trace[1]['node_id']}) in order",
            )
            check(
                t <= matching_trace[2],
                f"node_started for {data['node_id']} arrived at or before its own trace (completion) event",
            )

        # The property that actually matters overall: trace events arrived
        # BEFORE the done event, spread over time — not all bundled into
        # one final response. (In this sandbox, execution is fast enough
        # that timestamps may be very close together, but they must still
        # be monotonically non-decreasing and the trace events must
        # precede done, which wouldn't hold at all for a synchronous /run.)
        all_times = [t for _, _, t in received]
        check(all_times == sorted(all_times), "events arrived in non-decreasing wall-clock order (streamed, not batched)")
        check(trace_events[-1][2] <= done_events[0][2], "last trace event arrived at or before the done event")

        thread_id = done_events[0][1]["thread_id"]
        print(f"\n=== Streaming /resume (thread_id={thread_id}) ===")
        received_2 = []
        with httpx.stream(
            "POST",
            f"{BASE_URL}/agents/agt_stream_test/resume/stream",
            json={"thread_id": thread_id, "resume": {"approved": True}},
            timeout=20.0,
        ) as response:
            check(response.status_code == 200, f"resume stream should 200: {response.status_code}")
            for name, data, t in parse_sse_stream(response):
                received_2.append((name, data, t))
                if name == "trace":
                    print(f"  [trace] {data['node_label']}: {data['detail']}")
                elif name == "done":
                    print(f"  [done] status={data['status']}, email_sent={data['variables'].get('email_sent')}")

        resume_trace = [e for e in received_2 if e[0] == "trace"]
        resume_done = [e for e in received_2 if e[0] == "done"]
        check(len(resume_trace) >= 1, "resume should stream at least the remaining nodes (human_approval, send_email)")
        # The regression this specifically guards against: a resumed run's
        # checkpointed state already contains the FULL prior trace (all 5
        # nodes from before the pause), and graph.stream() re-yields that
        # whole trace in its first chunk — a naive seen_trace_len=0 on
        # resume would re-emit all 5 old events as if they were new,
        # duplicating them in any client that appends trace events
        # (which useAgentRunStream.ts does). Exactly 2 new events
        # (human_approval, send_email) — not 7 — is the correct behavior.
        check(
            len(resume_trace) == 2,
            f"resume should emit ONLY the newly-executed events (2: human_approval, send_email), "
            f"not re-emit the 5 pre-pause events too — got {len(resume_trace)}",
        )
        resumed_node_ids = [e[1]["node_id"] for e in resume_trace]
        check(
            resumed_node_ids == ["n_approval", "n_send"],
            f"resume trace contains exactly the two new nodes in order, no duplicates — got {resumed_node_ids}",
        )
        check(resume_done[0][1]["status"] == "completed", "resume should complete")
        check(resume_done[0][1]["variables"].get("email_sent") is True, "email was sent after streamed approval")

        print(f"\n{passed} passed, {failed} failed")
        if failed:
            raise SystemExit(1)

    finally:
        server.kill()
        server.wait(timeout=10)


if __name__ == "__main__":
    main()
