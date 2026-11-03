"""
P2 "Control flow" -- regression tests for if_else, switch, loop, wait, and
retry.

if_else/switch/loop have a plain no-op executor (exec_passthrough in
backend/compiler/executors.py) -- but that's not a stub standing in for
missing behavior. Their real semantics live entirely in their OUTGOING
EDGES' conditions, evaluated by compiler.py's _make_router -- the exact
same mechanism every other node's conditional routing already uses.
LangGraph's own cycle support (with its recursion_limit as the backstop)
is what makes `loop` a real loop, not custom node-level iteration logic.

wait and retry, by contrast, WERE plain passthroughs with no real
implementation at all -- fixed alongside these tests: wait now genuinely
calls time.sleep() (backend/compiler/executors.py's exec_wait), and retry
now genuinely retries-with-backoff whatever node it's wired to
(backend/compiler/compiler.py's _execute_with_retry, applied via a
per-node retry_policy lookup in compile_agent).

Run:
    PYTHONPATH=. python backend/compiler/test_control_flow.py
"""
from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from pydantic import ValidationError

from backend.compiler.compiler import compile_agent
from backend.models.agent_spec import AgentSpecification, NodeType


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_if_else_branches_correctly() -> None:
    print("=== if_else: real branching, both directions ===")
    spec = AgentSpecification.model_validate(
        {
            "metadata": {"id": "agt_if_else", "name": "X", "status": "draft", "version": "0.0.1", "createdAt": "2026-01-01T00:00:00Z", "updatedAt": "2026-01-01T00:00:00Z"},
            "goal": "test if_else branching",
            "nodes": [
                {"id": "n_trigger", "type": "trigger_manual", "category": "trigger", "label": "Start"},
                {"id": "n_if", "type": "if_else", "category": "control", "label": "Check amount"},
                {"id": "n_high", "type": "ask_ai", "category": "ai", "label": "High path", "config": {"prompt": "x", "output_key": "path"}},
                {"id": "n_low", "type": "ask_ai", "category": "ai", "label": "Low path", "config": {"prompt": "x", "output_key": "path"}},
            ],
            "edges": [
                {"id": "e1", "source": "n_trigger", "target": "n_if"},
                {"id": "e2", "source": "n_if", "target": "n_high", "condition": {"expression": "amount > 1000", "kind": "deterministic"}},
                {"id": "e3", "source": "n_if", "target": "n_low", "condition": {"expression": "amount <= 1000", "kind": "deterministic"}},
            ],
        }
    )

    def fake_ai(node, spec, variables):
        return {"path": node.label}

    graph = compile_agent(spec, checkpointer=MemorySaver(), executor_overrides={NodeType.ask_ai: fake_ai})

    r1 = graph.invoke({"variables": {"amount": 5000}, "trace": []}, config={"configurable": {"thread_id": "if-high"}})
    check([e["node_id"] for e in r1["trace"]] == ["n_trigger", "n_if", "n_high"], f"unexpected path: {r1['trace']}")
    check(r1["variables"]["path"] == "High path", f"unexpected: {r1['variables']}")

    r2 = graph.invoke({"variables": {"amount": 50}, "trace": []}, config={"configurable": {"thread_id": "if-low"}})
    check([e["node_id"] for e in r2["trace"]] == ["n_trigger", "n_if", "n_low"], f"unexpected path: {r2['trace']}")
    check(r2["variables"]["path"] == "Low path", f"unexpected: {r2['variables']}")
    print("  OK: amount=5000 takes the 'high' branch, amount=50 takes the 'low' branch")


def test_switch_routes_to_the_matching_case() -> None:
    print("\n=== switch: real multi-way branching (3 cases) ===")
    spec = AgentSpecification.model_validate(
        {
            "metadata": {"id": "agt_switch", "name": "X", "status": "draft", "version": "0.0.1", "createdAt": "2026-01-01T00:00:00Z", "updatedAt": "2026-01-01T00:00:00Z"},
            "goal": "test switch branching",
            "nodes": [
                {"id": "n_trigger", "type": "trigger_manual", "category": "trigger", "label": "Start"},
                {"id": "n_switch", "type": "switch", "category": "control", "label": "Route by label"},
                {"id": "n_a", "type": "ask_ai", "category": "ai", "label": "A", "config": {"prompt": "x", "output_key": "path"}},
                {"id": "n_b", "type": "ask_ai", "category": "ai", "label": "B", "config": {"prompt": "x", "output_key": "path"}},
                {"id": "n_c", "type": "ask_ai", "category": "ai", "label": "C", "config": {"prompt": "x", "output_key": "path"}},
            ],
            "edges": [
                {"id": "e1", "source": "n_trigger", "target": "n_switch"},
                {"id": "e2", "source": "n_switch", "target": "n_a", "condition": {"expression": "label == 'a'", "kind": "deterministic"}},
                {"id": "e3", "source": "n_switch", "target": "n_b", "condition": {"expression": "label == 'b'", "kind": "deterministic"}},
                {"id": "e4", "source": "n_switch", "target": "n_c", "condition": {"expression": "label == 'c'", "kind": "deterministic"}},
            ],
        }
    )

    def fake_ai(node, spec, variables):
        return {"path": node.label}

    graph = compile_agent(spec, checkpointer=MemorySaver(), executor_overrides={NodeType.ask_ai: fake_ai})
    for label, expected_node in [("a", "n_a"), ("b", "n_b"), ("c", "n_c")]:
        r = graph.invoke({"variables": {"label": label}, "trace": []}, config={"configurable": {"thread_id": f"switch-{label}"}})
        check([e["node_id"] for e in r["trace"]] == ["n_trigger", "n_switch", expected_node], f"unexpected path for label={label}: {r['trace']}")
    print("  OK: each of 3 labels routes to its own distinct case")


def test_loop_iterates_and_exits() -> None:
    print("\n=== loop: genuinely iterates and exits at the right count ===")
    spec = AgentSpecification.model_validate(
        {
            "metadata": {"id": "agt_loop", "name": "X", "status": "draft", "version": "0.0.1", "createdAt": "2026-01-01T00:00:00Z", "updatedAt": "2026-01-01T00:00:00Z"},
            "goal": "test loop iteration",
            "nodes": [
                {"id": "n_trigger", "type": "trigger_manual", "category": "trigger", "label": "Start"},
                {"id": "n_loop", "type": "loop", "category": "control", "label": "Loop"},
                {"id": "n_body", "type": "ask_ai", "category": "ai", "label": "Increment", "config": {"prompt": "x", "output_key": "counter"}},
            ],
            "edges": [
                {"id": "e1", "source": "n_trigger", "target": "n_loop"},
                {"id": "e2", "source": "n_loop", "target": "n_body", "condition": {"expression": "counter < 3", "kind": "deterministic"}},
                {"id": "e3", "source": "n_body", "target": "n_loop"},
            ],
        }
    )

    def increment(node, spec, variables):
        return {"counter": variables.get("counter", 0) + 1}

    graph = compile_agent(spec, checkpointer=MemorySaver(), executor_overrides={NodeType.ask_ai: increment})
    r = graph.invoke({"variables": {"counter": 0}, "trace": []}, config={"configurable": {"thread_id": "loop-1"}})
    check(r["variables"]["counter"] == 3, f"expected the loop to run exactly 3 times, got counter={r['variables']['counter']}")
    node_sequence = [e["node_id"] for e in r["trace"]]
    check(
        node_sequence == ["n_trigger", "n_loop", "n_body", "n_loop", "n_body", "n_loop", "n_body", "n_loop"],
        f"unexpected iteration sequence: {node_sequence}",
    )
    print(f"  OK: loop body ran exactly 3 times and exited cleanly, trace={node_sequence}")


def test_loop_without_an_exit_condition_is_rejected() -> None:
    print("\n=== a 'loop' with no conditional exit edge is still rejected (uncontrolled-cycle check) ===")
    try:
        AgentSpecification.model_validate(
            {
                "metadata": {"id": "agt_bad_loop", "name": "X", "status": "draft", "version": "0.0.1", "createdAt": "2026-01-01T00:00:00Z", "updatedAt": "2026-01-01T00:00:00Z"},
                "goal": "an uncontrolled loop should be rejected",
                "nodes": [
                    {"id": "n_trigger", "type": "trigger_manual", "category": "trigger", "label": "Start"},
                    {"id": "n_loop", "type": "loop", "category": "control", "label": "Loop"},
                    {"id": "n_body", "type": "ask_ai", "category": "ai", "label": "Body", "config": {"prompt": "x"}},
                ],
                "edges": [
                    {"id": "e1", "source": "n_trigger", "target": "n_loop"},
                    {"id": "e2", "source": "n_loop", "target": "n_body"},
                    {"id": "e3", "source": "n_body", "target": "n_loop"},
                ],
            }
        )
        raise AssertionError("a loop with no conditional exit edge should be rejected at spec-validation time")
    except ValidationError as exc:
        check("cycle" in str(exc).lower(), f"unexpected error: {exc}")
        print("  OK: an uncontrolled loop (no exit condition) is rejected before it could ever run forever")


def test_wait_genuinely_delays() -> None:
    print("\n=== wait: genuinely delays execution (not a no-op) ===")
    import time as time_module

    spec = AgentSpecification.model_validate(
        {
            "metadata": {"id": "agt_wait", "name": "X", "status": "draft", "version": "0.0.1", "createdAt": "2026-01-01T00:00:00Z", "updatedAt": "2026-01-01T00:00:00Z"},
            "goal": "test real wait",
            "nodes": [
                {"id": "n_trigger", "type": "trigger_manual", "category": "trigger", "label": "Start"},
                {"id": "n_wait", "type": "wait", "category": "control", "label": "Wait", "config": {"seconds": 0.3}},
            ],
            "edges": [{"id": "e1", "source": "n_trigger", "target": "n_wait"}],
        }
    )
    graph = compile_agent(spec, checkpointer=MemorySaver())
    start = time_module.monotonic()
    graph.invoke({"variables": {}, "trace": []}, config={"configurable": {"thread_id": "wait-test"}})
    elapsed = time_module.monotonic() - start
    check(elapsed >= 0.25, f"wait should have genuinely delayed execution by ~0.3s, only took {elapsed:.3f}s")
    print(f"  OK: a wait node with config.seconds=0.3 genuinely delayed the run by {elapsed:.2f}s")


def test_wait_reads_seconds_from_a_variable() -> None:
    print("\n=== wait: config.seconds can reference a variable by name ===")
    import time as time_module

    spec = AgentSpecification.model_validate(
        {
            "metadata": {"id": "agt_wait_var", "name": "X", "status": "draft", "version": "0.0.1", "createdAt": "2026-01-01T00:00:00Z", "updatedAt": "2026-01-01T00:00:00Z"},
            "goal": "test wait with a variable-driven duration",
            "nodes": [
                {"id": "n_trigger", "type": "trigger_manual", "category": "trigger", "label": "Start"},
                {"id": "n_wait", "type": "wait", "category": "control", "label": "Wait", "config": {"seconds": "delay"}},
            ],
            "edges": [{"id": "e1", "source": "n_trigger", "target": "n_wait"}],
        }
    )
    graph = compile_agent(spec, checkpointer=MemorySaver())
    start = time_module.monotonic()
    graph.invoke({"variables": {"delay": 0.3}, "trace": []}, config={"configurable": {"thread_id": "wait-var-test"}})
    elapsed = time_module.monotonic() - start
    check(elapsed >= 0.25, f"wait should have read 'delay'=0.3 from variables, only took {elapsed:.3f}s")
    print(f"  OK: a wait node with config.seconds='delay' correctly read 0.3 from variables, delayed by {elapsed:.2f}s")


def test_retry_recovers_from_transient_failures() -> None:
    print("\n=== retry: a node downstream of 'retry' genuinely retries on failure ===")
    spec = AgentSpecification.model_validate(
        {
            "metadata": {"id": "agt_retry_recover", "name": "X", "status": "draft", "version": "0.0.1", "createdAt": "2026-01-01T00:00:00Z", "updatedAt": "2026-01-01T00:00:00Z"},
            "goal": "test retry semantics",
            "nodes": [
                {"id": "n_trigger", "type": "trigger_manual", "category": "trigger", "label": "Start"},
                {"id": "n_retry", "type": "retry", "category": "control", "label": "Retry", "config": {"maxRetries": 3, "backoffSeconds": 0}},
                {"id": "n_flaky", "type": "api_call", "category": "tool", "label": "Flaky call"},
            ],
            "edges": [
                {"id": "e1", "source": "n_trigger", "target": "n_retry"},
                {"id": "e2", "source": "n_retry", "target": "n_flaky"},
            ],
        }
    )
    attempts = {"n": 0}

    def flaky(node, spec, variables):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ConnectionError(f"simulated transient failure (attempt {attempts['n']})")
        return {"result": f"succeeded on attempt {attempts['n']}"}

    graph = compile_agent(spec, checkpointer=MemorySaver(), executor_overrides={NodeType.api_call: flaky})
    r = graph.invoke({"variables": {}, "trace": []}, config={"configurable": {"thread_id": "retry-recover"}})
    check(attempts["n"] == 3, f"expected exactly 3 attempts, got {attempts['n']}")
    check(r["variables"]["result"] == "succeeded on attempt 3", f"unexpected: {r['variables']}")
    print(f"  OK: failed twice, succeeded on the 3rd attempt, exactly as configured (maxRetries=3)")


def test_retry_exhausts_and_raises() -> None:
    print("\n=== retry: exhausting all attempts still raises a real failure ===")
    spec = AgentSpecification.model_validate(
        {
            "metadata": {"id": "agt_retry_exhaust", "name": "X", "status": "draft", "version": "0.0.1", "createdAt": "2026-01-01T00:00:00Z", "updatedAt": "2026-01-01T00:00:00Z"},
            "goal": "test retry exhaustion",
            "nodes": [
                {"id": "n_trigger", "type": "trigger_manual", "category": "trigger", "label": "Start"},
                {"id": "n_retry", "type": "retry", "category": "control", "label": "Retry", "config": {"maxRetries": 2, "backoffSeconds": 0}},
                {"id": "n_always_fails", "type": "api_call", "category": "tool", "label": "Always fails"},
            ],
            "edges": [
                {"id": "e1", "source": "n_trigger", "target": "n_retry"},
                {"id": "e2", "source": "n_retry", "target": "n_always_fails"},
            ],
        }
    )
    attempts = {"n": 0}

    def always_fails(node, spec, variables):
        attempts["n"] += 1
        raise ConnectionError("permanent failure")

    graph = compile_agent(spec, checkpointer=MemorySaver(), executor_overrides={NodeType.api_call: always_fails})
    try:
        graph.invoke({"variables": {}, "trace": []}, config={"configurable": {"thread_id": "retry-exhaust"}})
        raise AssertionError("should have raised after exhausting all retry attempts")
    except RuntimeError as e:
        check(attempts["n"] == 3, f"expected exactly maxRetries+1=3 attempts, got {attempts['n']}")
        check("permanent failure" in str(e), f"unexpected error: {e}")
        print(f"  OK: made exactly 3 attempts (maxRetries=2 + 1 initial), then raised: {e}")


def test_node_without_a_retry_predecessor_gets_no_retry() -> None:
    print("\n=== a node with NO retry node upstream gets exactly one attempt (no behavior change) ===")
    spec = AgentSpecification.model_validate(
        {
            "metadata": {"id": "agt_no_retry", "name": "X", "status": "draft", "version": "0.0.1", "createdAt": "2026-01-01T00:00:00Z", "updatedAt": "2026-01-01T00:00:00Z"},
            "goal": "test that retry is opt-in via an upstream retry node",
            "nodes": [
                {"id": "n_trigger", "type": "trigger_manual", "category": "trigger", "label": "Start"},
                {"id": "n_fails_once", "type": "api_call", "category": "tool", "label": "Fails"},
            ],
            "edges": [{"id": "e1", "source": "n_trigger", "target": "n_fails_once"}],
        }
    )
    attempts = {"n": 0}

    def fails(node, spec, variables):
        attempts["n"] += 1
        raise ConnectionError("fails immediately")

    graph = compile_agent(spec, checkpointer=MemorySaver(), executor_overrides={NodeType.api_call: fails})
    try:
        graph.invoke({"variables": {}, "trace": []}, config={"configurable": {"thread_id": "no-retry"}})
        raise AssertionError("should have raised")
    except RuntimeError:
        check(attempts["n"] == 1, f"a node with no retry predecessor should get exactly 1 attempt, got {attempts['n']}")
        print("  OK: exactly 1 attempt made, retry is opt-in and doesn't change unrelated nodes' behavior")


def main() -> None:
    test_if_else_branches_correctly()
    test_switch_routes_to_the_matching_case()
    test_loop_iterates_and_exits()
    test_loop_without_an_exit_condition_is_rejected()
    test_wait_genuinely_delays()
    test_wait_reads_seconds_from_a_variable()
    test_retry_recovers_from_transient_failures()
    test_retry_exhausts_and_raises()
    test_node_without_a_retry_predecessor_gets_no_retry()
    print("\nAll control-flow checks passed: if_else/switch/loop (already real via edge routing), wait and retry (now genuinely implemented).")


if __name__ == "__main__":
    main()
