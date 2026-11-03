"""Sandboxed Python snippets for the Code node."""
from __future__ import annotations

import concurrent.futures
import math
import re
from typing import Any

from backend.compiler.executors import NodeExecutor
from backend.models.agent_spec import AgentNode, AgentSpecification


class CodeExecutionError(RuntimeError):
    pass


_ALLOWED_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "map": map,
    "range": range,
    "reversed": reversed,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
    "True": True,
    "False": False,
    "None": None,
}

_ALLOWED_MODULES = {
    "math": math,
    "re": re,
    "json": __import__("json"),
    "datetime": __import__("datetime"),
}


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name in _ALLOWED_MODULES:
        return _ALLOWED_MODULES[name]
    root = name.split(".")[0]
    if root in _ALLOWED_MODULES:
        return _ALLOWED_MODULES[root]
    raise CodeExecutionError(f"Import of '{name}' is not allowed in Code nodes.")


def _run_python(code: str, variables: dict[str, Any], timeout_sec: float = 5.0) -> dict[str, Any]:
    if not code or not str(code).strip():
        raise CodeExecutionError("Code node has empty source.")

    lines = code.strip().splitlines() or ["pass"]
    indented = "\n".join("    " + line for line in lines)
    if not any(line.strip().startswith("return ") for line in lines):
        indented += "\n    return {}"
    wrapped = "def __user_main(input):\n" + indented + "\n"

    safe_builtins = dict(_ALLOWED_BUILTINS)
    safe_builtins["__import__"] = _safe_import
    globals_dict: dict[str, Any] = {"__builtins__": safe_builtins}
    locals_dict: dict[str, Any] = {}

    def _execute():
        exec(wrapped, globals_dict, locals_dict)  # noqa: S102
        fn = locals_dict.get("__user_main") or globals_dict.get("__user_main")
        if fn is None:
            raise CodeExecutionError("Internal error: compiled user function missing.")
        out = fn(dict(variables))
        if out is None:
            return {}
        if not isinstance(out, dict):
            return {"result": out}
        return out

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(_execute)
        try:
            return fut.result(timeout=timeout_sec)
        except concurrent.futures.TimeoutError as exc:
            raise CodeExecutionError(f"Code node timed out after {timeout_sec}s.") from exc
        except CodeExecutionError:
            raise
        except Exception as exc:
            raise CodeExecutionError(f"Code node failed: {exc}") from exc


def build_code_executor() -> NodeExecutor:
    def execute(node: AgentNode, spec: AgentSpecification, variables: dict[str, Any]) -> dict:
        language = (node.config.get("language") or "python").lower()
        if language not in ("python", "py"):
            raise CodeExecutionError(f"Unsupported language '{language}'. Only python is supported.")
        source = node.config.get("code") or node.config.get("source") or ""
        timeout = float(node.config.get("timeout_sec") or 5)
        timeout = max(0.5, min(timeout, 30.0))
        result = _run_python(str(source), variables, timeout_sec=timeout)
        return {**result, "code_ok": True}

    return execute
