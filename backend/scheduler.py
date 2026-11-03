"""Simple interval scheduler for trigger_schedule nodes.

Runs inside the API process. Each workflow whose graph contains a
`trigger_schedule` node with config.enabled=true is invoked every
`interval_minutes` (minimum 1).

Not a full cron system — good enough for local inbox polling.
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Any, Callable

logger = logging.getLogger("agent_platform.scheduler")

_stop = threading.Event()
_thread: threading.Thread | None = None
_last_run: dict[str, float] = {}
_lock = threading.Lock()


def _schedule_config(spec) -> dict[str, Any] | None:
    for node in spec.nodes:
        if getattr(node, "type", None) is None:
            continue
        ntype = node.type.value if hasattr(node.type, "value") else str(node.type)
        if ntype != "trigger_schedule":
            continue
        cfg = node.config or {}
        if cfg.get("enabled") is False:
            return None
        try:
            minutes = float(cfg.get("interval_minutes") or cfg.get("interval") or 5)
        except (TypeError, ValueError):
            minutes = 5.0
        minutes = max(1.0, minutes)
        return {
            "interval_minutes": minutes,
            "node_id": node.id,
            "label": node.label,
        }
    return None


def _tick(store, run_fn: Callable[[str], None]) -> None:
    try:
        agents = store.list_agents()
    except Exception as e:
        logger.warning("scheduler list_agents failed: %s", e)
        return
    now = time.time()
    for spec in agents:
        try:
            agent_id = spec.metadata.id
            sched = _schedule_config(spec)
            if not sched:
                continue
            interval_sec = sched["interval_minutes"] * 60.0
            with _lock:
                last = _last_run.get(agent_id, 0.0)
                if now - last < interval_sec:
                    continue
                _last_run[agent_id] = now
            logger.info(
                "scheduler running agent=%s every %sm (node %s)",
                agent_id,
                sched["interval_minutes"],
                sched["node_id"],
            )
            try:
                run_fn(agent_id)
            except Exception as e:
                logger.exception("scheduler run failed for %s: %s", agent_id, e)
        except Exception as e:
            logger.warning("scheduler skip agent: %s", e)


def _loop(store_factory, run_fn: Callable[[str], None], poll_seconds: float = 15.0) -> None:
    logger.info("scheduler started (poll every %ss)", poll_seconds)
    while not _stop.is_set():
        try:
            store = store_factory()
            _tick(store, run_fn)
        except Exception as e:
            logger.warning("scheduler tick error: %s", e)
        _stop.wait(poll_seconds)
    logger.info("scheduler stopped")


def start_scheduler(store_factory, run_fn: Callable[[str], None]) -> None:
    global _thread
    if _thread and _thread.is_alive():
        return
    _stop.clear()
    _thread = threading.Thread(
        target=_loop,
        args=(store_factory, run_fn),
        name="agent-scheduler",
        daemon=True,
    )
    _thread.start()


def stop_scheduler() -> None:
    _stop.set()
