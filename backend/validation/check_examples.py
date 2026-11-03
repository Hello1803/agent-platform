"""
Cross-validates every spec in /examples against BOTH:
  1. schema/agent-spec.schema.json  (structural JSON Schema)
  2. backend/models/agent_spec.py   (Pydantic model + graph-integrity rules)

Run this in CI on every PR that touches the schema, the TS types, or the
Pydantic models — it's the cheapest way to catch the three representations
drifting out of sync.

Usage:
    python backend/validation/check_examples.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jsonschema
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.models.agent_spec import AgentSpecification  # noqa: E402

SCHEMA_PATH = ROOT / "schema" / "agent-spec.schema.json"
EXAMPLES_DIR = ROOT / "examples"


def main() -> int:
    schema = json.loads(SCHEMA_PATH.read_text())
    validator = jsonschema.Draft202012Validator(schema)

    example_files = sorted(EXAMPLES_DIR.glob("*.json"))
    if not example_files:
        print(f"No example specs found in {EXAMPLES_DIR}")
        return 1

    failures = 0

    for path in example_files:
        print(f"\n--- {path.name} ---")
        data = json.loads(path.read_text())

        schema_errors = sorted(validator.iter_errors(data), key=lambda e: e.path)
        if schema_errors:
            failures += 1
            print("  JSON Schema: FAIL")
            for err in schema_errors:
                loc = "/".join(str(p) for p in err.path) or "<root>"
                print(f"    - {loc}: {err.message}")
        else:
            print("  JSON Schema: OK")

        try:
            spec = AgentSpecification.model_validate(data)
            print(f"  Pydantic:    OK  ({len(spec.nodes)} nodes, {len(spec.edges)} edges)")
        except ValidationError as e:
            failures += 1
            print("  Pydantic:    FAIL")
            for err in e.errors():
                loc = ".".join(str(p) for p in err["loc"]) or "<root>"
                print(f"    - {loc}: {err['msg']}")

    print(f"\n{'=' * 40}")
    if failures:
        print(f"FAILED: {failures} issue(s) across {len(example_files)} example spec(s).")
        return 1

    print(f"PASSED: all {len(example_files)} example spec(s) valid against schema + models.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
