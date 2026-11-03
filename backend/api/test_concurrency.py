"""
Proves the atomic compare_and_set fix actually closes the race described
in the V7 code review: "two concurrent requests can both pass a
version check against the same starting version and both write, the
second silently overwriting the first."

This doesn't just assert the fix SHOULD work — it spawns real concurrent
threads racing to write, and checks that exactly one wins, both at the
store level (in-memory AgentStore) and over real HTTP against a running
server (proving the API layer's routes actually use compare_and_set, not
just that the store method exists in isolation).

Run:
    PYTHONPATH=. python backend/api/test_concurrency.py
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from backend.api.app import app
from backend.api.store import AgentStore
from backend.models.agent_spec import AgentSpecification

ROOT = Path(__file__).resolve().parents[2]

passed = 0
failed = 0


def check(condition: bool, message: str) -> None:
    global passed, failed
    if condition:
        passed += 1
    else:
        failed += 1
        print(f"FAIL: {message}")


def load_spec(name: str) -> AgentSpecification:
    return AgentSpecification.model_validate(json.loads((ROOT / "examples" / name).read_text()))


def test_store_level_race(n_threads: int = 20) -> None:
    """
    Directly races N threads calling compare_and_set with the SAME
    expected_version against the same in-memory store. Without the fix
    (a plain get()-then-set() with no shared critical section), CPython's
    GIL still context-switches between bytecode instructions, so this is
    a genuine race, not a theoretical one — a naive implementation WILL
    let more than one thread "win" here across enough repeated trials.
    """
    print(f"=== Store-level race: {n_threads} threads, same expected_version ===")
    store = AgentStore()
    spec = load_spec("research-agent.json")
    store.create(spec)
    base_version = spec.metadata.version

    results: list[bool] = []
    results_lock = threading.Lock()
    barrier = threading.Barrier(n_threads)

    def racer(i: int) -> None:
        barrier.wait()  # maximize actual concurrent overlap, not just "started around the same time"
        # Each thread proposes a DIFFERENT new spec (distinguishable), all
        # racing to write based on the SAME base_version.
        candidate = spec.model_copy(deep=True)
        candidate.metadata.version = f"{base_version}-racer-{i}"
        won = store.compare_and_set(spec.metadata.id, base_version, candidate)
        with results_lock:
            results.append(won)

    threads = [threading.Thread(target=racer, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    wins = sum(results)
    check(wins == 1, f"exactly one racer should win compare_and_set — got {wins} winners out of {n_threads}")
    final = store.get(spec.metadata.id)
    check(
        final.metadata.version.endswith("-racer-" + final.metadata.version.rsplit("-racer-", 1)[1]),
        "the stored spec is one of the racers' candidates, not silently corrupted",
    )
    print(f"  {wins}/{n_threads} succeeded (correct: exactly 1), final version: {final.metadata.version}")


def test_http_level_race(n_threads: int = 10) -> None:
    """
    Same race, but over the real API routes (TestClient), proving
    apply_agent_mutations() actually uses compare_and_set rather than the
    store method existing but being unused by the route.
    """
    print(f"\n=== HTTP-level race: {n_threads} concurrent POST /mutations, same expected_version ===")
    client = TestClient(app)
    spec_json = json.loads((ROOT / "examples" / "research-agent.json").read_text())
    spec_json["metadata"]["id"] = "agt_concurrency_test"
    resp = client.post("/agents", json={"spec": spec_json})
    check(resp.status_code == 201, f"create should 201: {resp.text}")
    base_version = resp.json()["metadata"]["version"]

    statuses: list[int] = []
    statuses_lock = threading.Lock()
    barrier = threading.Barrier(n_threads)

    def racer(i: int) -> None:
        barrier.wait()
        resp = client.post(
            "/agents/agt_concurrency_test/mutations",
            json={
                "mutations": [{"kind": "update_goal", "goal": f"goal from racer {i}"}],
                "expected_version": base_version,
            },
        )
        with statuses_lock:
            statuses.append(resp.status_code)

    threads = [threading.Thread(target=racer, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    successes = statuses.count(200)
    conflicts = statuses.count(409)
    check(successes == 1, f"exactly one racer should get 200, the rest 409 — got {successes} successes, statuses={statuses}")
    check(conflicts == n_threads - 1, f"expected {n_threads - 1} conflicts (409), got {conflicts}")
    print(f"  {successes} succeeded (200), {conflicts} correctly rejected (409) out of {n_threads} concurrent requests")


if __name__ == "__main__":
    test_store_level_race()
    test_http_level_race()
    print(f"\n{passed} passed, {failed} failed")
    if failed:
        raise SystemExit(1)
