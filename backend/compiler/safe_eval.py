"""
A deliberately small, restricted expression evaluator for edge conditions
like "amount > 10000" or "label == 'invoice' and confidence > 0.8".

Conditions in an AgentSpecification are authored by an LLM (via the NL
compiler) or a non-technical user (via the canvas), so we never hand them to
Python's real eval/exec. This walks a parsed AST and only permits a small,
safe subset: comparisons, boolean and/or/not, literals, and name lookups
resolved against the run's variable bag.

If a spec ever needs more expressive conditions than this supports, that's a
signal to add a specific, still-restricted AST node type here — not to fall
back to eval().
"""

from __future__ import annotations

import ast
import re
from typing import Any

_BOOL_NULL_RE = re.compile(r"\b(true|false|null)\b", re.IGNORECASE)
_BOOL_NULL_MAP = {"true": "True", "false": "False", "null": "None"}

_ALLOWED_COMPARE_OPS = (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn)
_ALLOWED_BOOL_OPS = (ast.And, ast.Or)


class UnsafeExpressionError(ValueError):
    pass


def _normalize_literals(expr: str) -> str:
    """Allow JS/JSON-style true/false/null in author-written expressions."""
    return _BOOL_NULL_RE.sub(lambda m: _BOOL_NULL_MAP[m.group(1).lower()], expr)


def _eval_node(node: ast.AST, context: dict[str, Any]) -> Any:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, context)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (str, int, float, bool)) or node.value is None:
            return node.value
        raise UnsafeExpressionError(f"Unsupported constant type: {type(node.value)}")

    if isinstance(node, ast.Name):
        if node.id not in context:
            raise UnsafeExpressionError(f"Unknown variable '{node.id}' in condition.")
        return context[node.id]

    if isinstance(node, ast.Compare):
        left = _eval_node(node.left, context)
        result = True
        for op, comparator in zip(node.ops, node.comparators):
            if not isinstance(op, _ALLOWED_COMPARE_OPS):
                raise UnsafeExpressionError(f"Operator {type(op).__name__} not allowed.")
            right = _eval_node(comparator, context)
            if isinstance(op, ast.Eq):
                ok = left == right
            elif isinstance(op, ast.NotEq):
                ok = left != right
            elif isinstance(op, ast.Lt):
                ok = left < right
            elif isinstance(op, ast.LtE):
                ok = left <= right
            elif isinstance(op, ast.Gt):
                ok = left > right
            elif isinstance(op, ast.GtE):
                ok = left >= right
            elif isinstance(op, ast.In):
                ok = left in right
            elif isinstance(op, ast.NotIn):
                ok = left not in right
            else:  # pragma: no cover - guarded above
                raise UnsafeExpressionError(f"Operator {type(op).__name__} not allowed.")
            result = result and ok
            left = right
        return result

    if isinstance(node, ast.BoolOp):
        if not isinstance(node.op, _ALLOWED_BOOL_OPS):
            raise UnsafeExpressionError(f"Boolean operator {type(node.op).__name__} not allowed.")
        values = [_eval_node(v, context) for v in node.values]
        return all(values) if isinstance(node.op, ast.And) else any(values)

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return not _eval_node(node.operand, context)

    raise UnsafeExpressionError(f"Expression node type {type(node).__name__} not allowed.")


def safe_eval(expression: str, context: dict[str, Any]) -> bool:
    """
    Evaluate a restricted boolean expression against a variable context.
    Raises UnsafeExpressionError for anything outside the allowed grammar,
    and for references to undefined variables (fail loud, not silently
    False — a typo'd variable name in a spec should surface at compile/run
    time, not silently misroute the agent).
    """
    normalized = _normalize_literals(expression)
    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError as e:
        raise UnsafeExpressionError(f"Could not parse condition '{expression}': {e}") from e
    return bool(_eval_node(tree, context))
