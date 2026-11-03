# Agent Specification and Implementation Notes


# Agent Specification (IR) — v1.0.0

This is the core artifact of the platform: the single source of truth that
the natural-language builder, the visual canvas, and the runtime compiler
all read and write. Nothing else in the system should hold its own copy of
"what the agent does."

## Layout

```
schema/agent-spec.schema.json        canonical JSON Schema (draft 2020-12) — the real spec
packages/shared-types/src/agent-spec.ts   TypeScript types for the frontend/canvas
backend/models/agent_spec.py         Pydantic models for the compiler/runtime
backend/validation/check_examples.py cross-checks example specs against schema + models
examples/invoice-agent.json          worked example: email → extract → approve → send
examples/research-agent.json         worked example: simple search → summarize agent
```

## Why three representations of the same thing?

- **JSON Schema** is the contract. It's language-agnostic, so it's what you'd
  hand to any future service (or an LLM being asked to *generate* a spec) as
  the ground truth of what's valid.
- **TypeScript types** let the React canvas get compile-time safety while
  rendering/editing nodes and edges, and while sending `AgentSpecMutation`
  objects up to the backend.
- **Pydantic models** are where the real enforcement happens on the backend —
  they include checks the JSON Schema can't express (dangling edge
  references, tool refs pointing at nothing, approval rules pointing at
  nodes that don't exist). This is your Agent Compiler's validation layer
  from section 7 of the architecture doc.

Run `python backend/validation/check_examples.py` any time you touch any of
the three files — it's the tripwire for drift between them. Wire it into CI
before this grows past one contributor.

## Key design decisions baked into the schema

- **`executionType: deterministic | probabilistic`** on every node and edge
  condition — this is what lets the canvas visually distinguish `[IF]` logic
  from `[AI DECISION]` blocks (architecture doc §11). The Pydantic model
  additionally *enforces* that only node types capable of making a model
  call can be marked probabilistic, so the compiler catches a
  spec-generation bug instead of silently mislabeling a hard-coded branch.
- **`AgentSpecMutation`** (in both TS and Python) is the only sanctioned way
  either interface writes to a spec. Chat and canvas both emit mutations;
  a single Workflow Mutation Engine applies + validates + recompiles. This
  is what makes concurrent chat-edit + drag-edit safe (§23, "Concurrent
  Modifications").
- **`toolRefs` + `ToolBinding.permissions`** separate *which tools exist for
  this agent* from *which node is allowed to call them*, and credentials are
  referenced (`credentialRef`), never inlined — this is the hook for the
  security model in §19.
- **`subgraph`** on a node lets a published agent be dropped in as a node
  inside a larger workflow (§9, "Agent vs Workflow"), without needing a
  separate node-type taxonomy for "agent" vs "workflow."
- **`metadata.publishedVersion`** vs `metadata.version` models the
  draft/live split from §16 directly in the spec, rather than bolting
  versioning on as an external database concern later.

## What's deliberately NOT in this schema

- **Canvas layout beyond `canvasPosition`** — auto-layout, grouping, and
  collapsing (§23 "Visual Clutter") are presentation-layer concerns and
  should live in a separate `ui_layout` document keyed by node id, not in
  the spec that the compiler/runtime consumes. Keeping them out avoids
  invalidating execution semantics every time someone drags a node.
- **Execution state / trace data** — a run's trace (§12) is an output of
  executing a spec, not part of the spec itself. It'll want its own schema,
  keyed by `(agentId, version, runId)`.

## Compiler (implemented)

### Tool registry and binding modes

`backend/tools/registry.py` is the authoritative catalogue of executable
tool contracts. A node binding is now checked for a registered tool id, node
compatibility, capability, and—when `mode: "connected"`—the required
credential reference. Palette-created `unconfigured.*` placeholders are
deliberately invalid for Test/Run until the Inspector selects a compatible
tool. `mode: "demo"` is explicit: it is permitted only for the documented
demo/stub path and must not be represented as an external operation.

The compiler also blocks Test/Run for disconnected graph components. Build
mode can still contain incomplete work while a workflow is being designed.

`backend/compiler/` turns a validated `AgentSpecification` into a runnable
LangGraph graph:

- `state.py` — the shared `AgentState` (a `variables` data bag + an
  append-only `trace` log, both with LangGraph reducers so concurrent node
  updates merge instead of clobbering).
- `safe_eval.py` — a restricted AST-walking evaluator for edge conditions
  (`amount > 10000`, `label == 'invoice'`). Deliberately not real `eval()` —
  conditions are authored by an LLM or a non-technical user, so the grammar
  is limited to comparisons/boolean ops/literals/variable lookups, and
  anything else raises rather than silently running.
- `executors.py` — one stub function per `NodeType`, all sharing the same
  `(node, spec, variables) -> dict` signature. These are where real LLM/tool
  calls plug in later without touching the compiler. `human_approval`,
  `ask_user`, `confirm`, and `review` use LangGraph's `interrupt()` to
  actually pause the graph and hand the resume value back into state —
  this is the real mechanism behind the approval cards in doc section 13,
  not a placeholder.
- `compiler.py` — `compile_agent(spec)` builds the `StateGraph`: one graph
  node per spec node, `START` wired to every node with no incoming edge,
  `END` wired from every node with no outgoing edge, and conditional edges
  routed through `safe_eval` in priority order. It also runs two
  compile-time checks the Pydantic model can't express: every node that
  calls a permissioned tool (`send_email`, `database`, etc.) must have a
  `toolRefs` binding that actually grants the needed capability, and no
  tool binding may use a `tool_id` that's on `spec.permissions.deny`.

Run `PYTHONPATH=. python backend/run_examples.py` to see it end-to-end: the
research agent runs straight through, and the invoice agent pauses at
`human_approval`, resumes with an `Command(resume=...)` approval decision,
and — critically — a **rejected** approval is verified to *not* reach the
`send_email` node. That's not cosmetic; it's the edge condition
(`approved == true`) actually gating tool access.

## Mutation Engine (implemented)

`backend/mutation/engine.py` is the only sanctioned way either interface
changes a spec — the chat builder and the canvas both emit
`AgentSpecMutation` objects (now 14 kinds, kept 1:1 between
`backend/models/agent_spec.py` and `packages/shared-types/src/agent-spec.ts`,
parsed on the Python side via a `kind`-discriminated `TypeAdapter`).

Two properties this gives you, both demonstrated in
`backend/run_mutations.py`:

- **Atomic batches** — `apply_mutations(spec, [...])` applies a list as one
  unit. If mutation 3 of 5 is invalid (e.g. an edge pointing at a node that
  doesn't exist yet), mutations 1–2 are rolled back too — useful because
  the NL compiler will often need to add a node *and* wire an edge to it in
  the same turn.
- **Optimistic concurrency** — every batch can carry the `metadata.version`
  it was computed against. If the spec has already moved past that version
  by the time the batch is applied, it's rejected with an explicit
  conflict instead of silently overwriting someone else's edit. This is
  the concrete mechanism behind doc section 23's "Concurrent
  Modifications" problem (the canvas-drag-vs-chat-edit race) —
  `run_mutations.py`'s last demo plays out exactly that race and confirms
  neither edit is lost.

`remove_node` and `remove_tool` also cascade — deleting a node cleans up
edges and approval rules that pointed at it, deleting a tool binding strips
it from any node's `toolRefs`, so a canvas "delete" doesn't leave the spec
in a state graph-integrity validation would reject with an opaque error.

## NL Compiler (implemented)

`backend/nl_compiler/` turns a plain-language instruction into a validated,
applied spec change — feeding straight into the mutation engine above
rather than reimplementing validation:

- `schemas.py` — `MutationBatch` (`reasoning` + a discriminated list of
  `AgentSpecMutation`), the structured-output contract. The model never
  regenerates a whole spec — only the specific mutations an instruction
  implies, which is what keeps changes small, auditable, and safe to feed
  straight into the mutation engine's atomicity/versioning.
- `prompts.py` — grounds the model in the actual node-type catalog (so it
  can't invent a node type the platform doesn't support) and the **current**
  spec's nodes/edges/tools/approval rules (so "ask me before sending the
  email" resolves against the agent that actually exists). Also encodes the
  doc's ambiguity-handling guidance directly: prefer read-only /
  approval-gated interpretations when an instruction is genuinely
  ambiguous, rather than guessing at something irreversible.
- `compiler.py` — `compile_instruction(spec, message)` calls
  `client.messages.parse(output_format=MutationBatch)` (Claude's structured
  outputs — guarantees schema-valid JSON, no hand-rolled parsing/retries
  for malformed output), then hands the mutations to
  `mutation.engine.apply_mutations()`. Critically, structured outputs only
  guarantees the *shape* is valid — not that a referenced node id actually
  exists in *this* spec. When the mutation engine's validation catches
  that (a hallucinated or misremembered id), the compiler feeds the error
  back to the model and retries once before giving up — this is the
  concrete mitigation for doc section 23's "AI → Workflow Reliability"
  problem.

`backend/run_nl_compiler.py` exercises the whole pipeline against a fake
Anthropic client (no API key needed to verify the logic) covering: a clean
instruction that adds and wires a node, an instruction where the model
initially references a nonexistent node id and self-corrects on retry, and
a pure question that resolves to zero mutations without touching the spec's
version. Point `compile_instruction` at a real `Anthropic()` client (just
omit `client=`) to go live — nothing else changes.

## Canvas (implemented)

`packages/canvas/` is the React Flow visual editor over the same
`AgentSpecification` and `AgentSpecMutation` types the NL compiler uses —
same contract, so the canvas and chat builder are structurally incapable
of drifting into separate representations of the agent.

- `tokens.css` — the one deliberate design decision in this package
  operationalizes architecture doc section 11 directly: **executionType**
  controls border style (solid/sharp corners = deterministic, dashed +
  gradient-tinted = probabilistic) as an axis fully independent from
  **category** color (trigger=green, ai=violet, tool=blue,
  control=charcoal, human=amber). A probabilistic `ai` node and a
  deterministic `tool` node are never confusable at a glance, and neither
  are two different probabilistic nodes of different categories.
- `specToFlow.ts` — pure, framework-light conversion between
  `AgentSpecification` and React Flow's node/edge shapes. Kept free of
  React so it's unit-testable without a browser (`__tests__/specToFlow.test.ts`,
  16 assertions, run via `tsx`) — verified against the real
  `invoice-agent.json` spec: all 3 AI nodes correctly flagged
  `probabilistic`, the `label == 'invoice'` branch condition correctly
  carried through to the edge renderer.
- `nodes/AgentNodeCard.tsx`, `edges/ConditionEdge.tsx` — the actual
  category/determinism rendering, and an edge label that surfaces the
  condition expression in monospace (with a distinct dashed style for
  `kind: "ai_decision"` conditions vs plain deterministic ones).
- `AgentCanvas.tsx` — the spec-is-truth pattern: local React Flow state
  gives smooth drag/select feedback, but the component never treats that
  local state as authoritative. A completed drag or a new connection
  becomes exactly one `AgentSpecMutation` handed to `onMutation` — the
  same shape the NL compiler produces — and the canvas resyncs from
  `spec` whenever `metadata.version` changes. This is what makes a
  **rejected** mutation (e.g. a version conflict from a concurrent chat
  edit) resolve correctly: the optimistic local change is simply
  overwritten by the next authoritative `spec` render, with no special-case
  rollback code needed.

`npm install && npm run typecheck --workspaces` typechecks both TS
packages together (canvas correctly resolves `@agent-platform/shared-types`
as an npm workspace). `npm run test:canvas` runs the pure-function suite.

## API Layer (implemented)

`backend/api/app.py` is a thin FastAPI service wiring everything above
into one runnable HTTP surface — this is what a real frontend (the canvas
package, or a mobile app, or `curl`) actually talks to:

```
POST /agents                     create an agent from a spec
GET  /agents/{id}                fetch the current spec
POST /agents/{id}/mutations      apply a mutation batch (canvas edits)
POST /agents/{id}/instructions   NL instruction -> mutations (chat builder)
POST /agents/{id}/compile        dry-run compile check ("Test" mode, doc section 14)
POST /agents/{id}/run            compile + execute, pausing on human_approval
POST /agents/{id}/resume         resume a paused run with an approval decision
POST /agents/{id}/publish        promote current draft to publishedVersion
```

Deliberately thin: every real decision (validation, atomicity, version
conflicts, graph compilation, permission checks) already lives in the
mutation engine, compiler, and NL compiler — this file is HTTP plumbing
only, mapping `MutationResult`/`CompileError`/version-conflicts to the
right status codes (409 for a concurrency conflict, 422 for a validation
failure, 404 for an unknown agent, 502 if the model itself refuses).

`backend/api/store.py` is an intentionally minimal in-memory `AgentStore`
(a dict behind a lock) — the point is to nail down the request/response
contracts before committing to a database. Swapping in real persistence
(Postgres for specs, a durable LangGraph checkpointer for run state) only
touches this one file.

`backend/api/test_app.py` exercises the **entire lifecycle over real HTTP
requests** via FastAPI's `TestClient` — create, duplicate-rejection, 404,
mutation apply, a stale-version 409, a malformed-mutation 422, an NL
instruction through a fake Anthropic client (same override pattern as
`run_nl_compiler.py`, via `app.dependency_overrides`), a compile check, a
full run that pauses for approval and resumes to completion with
`email_sent == True`, and finally publish. All 13 checks pass against the
real app object, not mocks of it.

One real bug this test caught and fixed: publishing was initially routed
through the generic mutation engine, which — correctly, for real edits —
always bumps `metadata.version` on success. But publishing is a status
transition on the *current* draft, not a new edit to it, so that bump
made `publishedVersion` chase a version number that had already moved past
it. Fixed with a dedicated `mutation.engine.publish()` that freezes
`publishedVersion` at whatever version was live at call time, matching
doc section 16's "Published: v1.4 / Draft: v1.5" model exactly.

## Canvas ↔ API wiring (implemented)

The "last piece of glue" from the previous section is done:

- `apiClient.ts` — a typed client for every route in `backend/api/app.py`.
  Mirrors the request/response schemas by hand (same cross-language pattern
  as the schema/TS-types/Pydantic trio at the top of this doc). `ApiError`
  carries the HTTP status and exposes `.isVersionConflict` /
  `.isValidationError` so callers can tell "someone else's edit landed
  first, re-fetch and retry" apart from "this specific mutation was wrong."
- `useAgentSession.ts` — the hook that actually wires `AgentCanvas`'s
  `onMutation` to the live API: owns the authoritative `spec`, calls
  `applyMutations` with `expected_version` on every mutation, and updates
  `spec` from the response. A request-sequence counter guards against a
  slow, stale response clobbering a newer one if the user fires two
  mutations quickly.
- **A real gap this surfaced and fixed**: `AgentCanvas`'s resync effect
  was keyed only on `spec.metadata.version`. For a *rejected* mutation,
  though, the server makes no change — version stays identical — so that
  effect would never re-fire, and the user's optimistic local drag would
  stay visually stuck in the wrong place instead of snapping back. Fixed
  by adding an explicit `resyncNonce` prop: `useAgentSession` bumps it on
  every failed `applyMutations` call (conflict or plain validation
  failure), forcing `AgentCanvas` to discard the optimistic change and
  redraw from the (unchanged) authoritative spec — no special-case
  rollback logic needed in the canvas itself.

`__tests__/apiClient.integration.test.ts` is the strongest verification
in this repo: it spawns the **real** `uvicorn` server as a subprocess and
drives it with the **real** `AgentApiClient` over genuine HTTP — no
`TestClient`, no mocks on either side. It creates an agent, applies a
mutation, deliberately triggers a version conflict and confirms it
surfaces as `ApiError.isVersionConflict` through the actual TS client
(not just in the Python-side test), runs the compile check and a full
run over HTTP, and publishes. 11/11 assertions pass against a genuinely
running process.

## Real Executors (implemented, partially)

`backend/real_executors/` replaces stub node executors with actual working
implementations — proving the executor interface `(node, spec, variables)
-> dict` really is enough to plug in something real without touching the
compiler, which took one small, backward-compatible change:
`compile_agent()` now accepts `executor_overrides`, merged over the
default stub registry per node type.

- `ai.py` — real Claude-backed executors for all 6 AI-category node types,
  using the same structured-outputs pattern as the NL compiler. The
  interesting problem: a `classify` node's labels, or an
  `extract_information` node's fields, are defined by the spec author (via
  canvas or NL compiler) at *spec-authoring* time, not code-writing time —
  so the structured-output schema has to be built **dynamically per node**,
  not hardcoded. `_build_classify_schema` constructs a `Literal[...]` type
  from `node.config.labels` and `_build_extract_schema` builds a
  per-field-typed Pydantic model from `node.config.fields` +
  `field_types`, both via `pydantic.create_model`. Verified this is
  genuinely dynamic (not coincidentally matching just the invoice
  example) by building schemas for a completely different sentiment-classify
  node and a mixed-type order-extraction node, confirming the classify
  schema actually *rejects* an out-of-domain label and the extract schema
  correctly types a `number` and a `boolean` field.
- `database.py` — a real SQLite-backed `database` executor. Unlike
  Gmail/Sheets, this needs no external credentials this sandbox can't
  provide, so it's the one tool node with a genuinely working
  implementation rather than a documented stub: it actually creates a
  table, inserts a row, and `read_records()` reads it back. Respects
  `node.config.fields` filtering (only persist the configured columns, not
  the whole variable bag) and re-checks the `write` permission scope at
  call time, same defense-in-depth pattern as the stub executors'
  `_require_tool`.

`backend/run_real_executors.py` runs the **real compiler and real
LangGraph execution** — including the real human-approval interrupt —
against the invoice-agent spec, with only the Anthropic client faked (no
API key in this sandbox; same fake-client pattern used everywhere else in
this repo). Confirms: the classify node's dynamic schema correctly returns
`label="invoice"` with `confidence=0.94`; the extract node's dynamic
per-field schema correctly types `amount` as a float; the free-text
`ai_agent` executor's draft email round-trips into `variables["draft_email"]`;
and — the part that isn't simulated at all — **a real SQLite file on disk
actually contains one row with exactly the three configured fields**,
filtered correctly from the full variable bag.

`send_email`, `read_email`, `search_web`, and the rest remain documented
stubs — they genuinely need external credentials (OAuth for Gmail,
a search API key) this environment has no way to provision. The
`ToolPermissionError` re-check pattern and the `NodeExecutor` signature
are identical between stub and real implementations, so wiring in a real
Gmail send later is exactly the same shape of change as `database.py` was.

## Persistence (implemented)

`backend/persistence/postgres_store.py` is a real, Postgres-backed
`AgentStore` — proven against an actual running local PostgreSQL 16
instance, not mocked. Getting there required a real architectural fix
first: `backend/api/app.py` originally imported its `store` as a bare
module-level singleton, so swapping in Postgres would have meant editing
every route. Refactored the store into a FastAPI dependency (`get_store`,
same pattern already used for `get_anthropic_client`) — every route now
takes `store: AgentStore = Depends(get_store)`, the default still resolves
to the in-memory store (zero setup, what every other test in this repo
uses), and a real deployment overrides it with one line via
`app.dependency_overrides[get_store]`.

Two tables matter here:
- `agents` — the current `AgentSpecification` per agent, as JSONB.
- `checkpoints` / `checkpoint_blobs` / `checkpoint_writes` — owned and
  managed entirely by `langgraph-checkpoint-postgres`'s `PostgresSaver`.
  This is the piece that actually matters: a `MemorySaver` checkpoint (the
  default everywhere else in this repo) is a process-local dict — a paused
  `human_approval` interrupt is gone the moment the server restarts. A
  `PostgresSaver` checkpoint is a row in a real database.

`backend/persistence/test_postgres_persistence.py` is the proof, and it's
deliberately not just "read the spec back from the same object." It:
1. Creates an agent and starts a run with **store instance #1**, pausing
   it at `human_approval`.
2. Explicitly closes and discards store instance #1 — no shared Python
   state, no shared connection, nothing carried over in memory.
3. Constructs a **brand-new** `PostgresAgentStore` (store instance #2, a
   fresh connection, a fresh `PostgresSaver`) pointed at the same
   database, and confirms the spec reads back correctly *and* the paused
   run resumes to completion (`email_sent == True`) — from a store object
   that has never seen that `thread_id` in its own process before.

This is the closest thing to "survives a server restart" that's provable
without literally restarting a process — two independent store instances
against the same real database is exactly what a restart looks like from
the checkpointer's point of view. Directly inspected the database
afterward too: `agents` held the persisted spec, and `checkpoints` held 9
real rows from that one run — not simulated, not an artifact of test
scaffolding.

Setting up local Postgres for this (if you don't already have it):
```bash
apt-get install -y postgresql postgresql-contrib
pg_ctlcluster 16 main start   # or whatever version apt installed
# enable TCP (Debian/Ubuntu's default only listens on the unix socket):
sed -i "s/^#listen_addresses = 'localhost'/listen_addresses = 'localhost'/" \
  /etc/postgresql/16/main/postgresql.conf
pg_ctlcluster 16 main restart
su postgres -c "psql -c \"CREATE USER agent_platform WITH PASSWORD 'agent_platform_dev';\""
su postgres -c "psql -c \"CREATE DATABASE agent_platform OWNER agent_platform;\""
```

### Wired into the actual running server

`backend/api/app.py`'s `get_store()` now reads `DATABASE_URL` directly: set
it and the server lazily constructs one process-wide `PostgresAgentStore`
(not one per request — its `PostgresSaver` does real DDL in `setup()`, not
something to repeat every call); leave it unset and every route falls back
to the in-memory store exactly as before, so nothing else in this repo's
test suite needed to change.

`backend/persistence/test_server_restart.py` is the proof at the level
that actually matters in production — not a dependency-override
simulation, a **literal OS process restart**: it starts the real `uvicorn`
server as a subprocess with `DATABASE_URL` set, creates an agent and pauses
a run on it over genuine HTTP, then `SIGKILL`s that process outright (no
graceful shutdown) and starts a brand-new `uvicorn` subprocess with a
different PID against the same database. The new process correctly reads
the spec back and **resumes the paused run to completion**
(`email_sent == True`) — a run that a process which no longer exists
started. This is the definitive version of what
`test_postgres_persistence.py` already proved one layer down (two
independent Python store objects); this proves it one layer up, where a
real deployment would actually feel it.

One real hiccup hit building this: the local Postgres server had stopped
between sessions in this sandbox (container reset, not a code bug) —
diagnosed via the test's captured subprocess logs (`_server1.log` /
`_server2.log`, written instead of `DEVNULL` specifically so a failure like
this wouldn't be a silent, undiagnosable timeout), restarted it, and
confirmed the actual table data survived on disk the whole time — which is
itself a small, honest demonstration of the exact property this section
is about.

## Execution Trace UI (implemented)

Doc section 12's "black box problem" gets a real answer here, both ends:

**Backend — SSE streaming.** `/agents/{id}/run/stream` and
`/agents/{id}/resume/stream` (`backend/api/app.py`) stream trace events as
Server-Sent Events using `graph.stream(..., stream_mode="values")` instead
of the synchronous `graph.invoke()` the plain `/run`/`/resume` routes use —
one `trace` SSE event per node as it actually finishes executing, then a
`done` event carrying the exact same shape the synchronous routes return.
`backend/api/test_streaming.py` proves this is genuinely incremental, not
batched-then-flushed: it captures wall-clock timestamps per SSE event and
asserts the trace events for a 5-node run arrive in order *before* the
final `done` event, against a real running server.

**Frontend — trace overlay + log.** Browser `EventSource` doesn't support
POST bodies (needed for `seed_variables`), so `sseClient.ts` implements SSE
parsing over `fetch`'s `ReadableStream` instead — a well-established
pattern for exactly this limitation. Its chunk parser is unit-tested
separately from the network plumbing, including the case that actually
matters: an SSE frame split across two `fetch` chunks (a `data:` line cut
mid-JSON) reassembles correctly.

- `applyTraceToNodes` (`specToFlow.ts`) is the pure function translating a
  trace list into per-node visual status — `AgentCanvas` now accepts
  `trace`/`activeNodeId` props and computes this as a derived overlay
  layer *on top of* its own drag/selection state, never writing into it,
  so streaming a trace can't fight with an in-progress drag.
  `AgentNodeCard` renders it as a status-colored ring (green/red/amber)
  and dims nodes that weren't on the executed path — but *only* once a
  trace has actually started (a separate `traceActive` flag), so the
  normal editing view isn't dimmed by default. Caught and fixed exactly
  this bug during development: my first version conflated "no trace status
  for this node" with "no trace happening at all," which would have dimmed
  every single node the moment the app loaded.
- `useAgentRunStream.ts` wires `AgentApiClient.runAgentStream`/
  `resumeAgentStream` (SSE versions of `runAgent`/`resumeAgent`) into React
  state a component can consume directly.
- `TraceLog.tsx` renders the doc's own example format almost verbatim —
  `✓ Email retrieved` / `⏸ Waiting for human approval` — as a clickable
  list for the "why did this execute?" affordance.

**One honest limitation from the previous iteration, now closed**: the
backend now emits a real `node_started` SSE event, sourced from
LangGraph's `custom` stream mode — every compiled node function takes an
injected `writer` parameter and calls it the instant it begins, *before*
its (possibly slow, possibly model-calling) executor runs. Verified this
interleaves genuinely live rather than being buffered until the node's
return value is ready: an isolated test with artificial `time.sleep`
delays showed `custom` writes and `values` snapshots arriving at
measurably different wall-clock timestamps, not bunched together.
`useAgentRunStream.ts` now actually populates `activeNodeId` from this —
previously an honestly-documented dead field. One more real thing this
surfaced in testing: the `human_approval` node gets a `node_started` but
*never* a matching `trace` completion event, since it pauses via
`interrupt()` instead of returning — which turns out to be exactly the
right behavior (`activeNodeId` correctly stays pointed at it while
paused, reading naturally as "still waiting on this node"), not a bug to
paper over. `test_streaming.py`'s assertions account for this precisely
(`len(started) == len(trace) + 1`, with the extra one identified as the
paused node) rather than assuming a naive 1:1 pairing.

## NL Builder Chat UI (implemented)

The doc's core concept — "natural language creates and modifies the
workflow; the visual workflow exposes and explains the agent" — finally
has both halves sharing state in one place:

- **`useAgentSession` now owns chat too**, not a separate hook. This was
  the one architectural decision that mattered here: a second hook with
  its own copy of `spec` would have reintroduced exactly the split-brain
  problem the whole "Single Source of Truth" section of the doc exists to
  prevent. Instead, `sendInstruction` and `applyMutation` write through to
  the identical `spec` state — type "ask me before sending the email" in
  chat, and the `human_approval` node appears on the canvas the same
  render, no separate sync step.
- **`chatMessages.ts`** is the pure logic — building a `ChatMessage` from
  a successful `InstructionResult` or a failed `ApiError` — deliberately
  factored out of the hook so it's unit-testable without React. Caught a
  real bug this way: the first version of `buildErrorMessage` leaked
  `ApiError`'s raw `"API error 502: ..."` prefix straight into
  user-facing chat text; a test asserting no raw status code appears in
  the message caught it immediately, fixed before it ever reached a
  screen.
- **`ChatPanel.tsx`** is deliberately presentational — takes `messages`/
  `onSend`/`busy` as props, no state of its own, so it's swappable or
  restylable without touching `useAgentSession`. Distinguishes a version
  conflict ("someone else's edit landed first, try again") from a
  validation failure (surfaces the actual error detail) from a model
  refusal, since the right next step differs for each.
- Both `applyMutation` and `sendInstruction` share one `requestSeq`
  counter, so a canvas drag and a chat instruction that both happen to be
  in flight at once can't have the slower one silently clobber the
  faster one's result — same stale-response guard, now covering both
  input sources at once instead of two independent ones.

52 pure-function assertions now cover the canvas package's logic (24
spec↔flow conversion, 12 SSE parsing, 16 chat message construction) — none
of them need a browser or a running server, by design, so they run in
milliseconds as part of `npm run test:canvas`.

## App Shell (implemented)

`packages/app/` is the page that finally puts every previously-built,
independently-tested piece on screen together — a real Vite + React app,
not a mockup:

- **Build mode**: `AgentCanvas` and `ChatPanel` side by side, both reading
  the *same* `useAgentSession` — this is the payoff of the "one hook owns
  both" decision from the chat UI section above. Drag a node, then type an
  instruction; both write through to one `spec`.
- **Test mode**: the same canvas in read-only mode with a live trace
  overlay (`trace`/`activeNodeId` from `useAgentRunStream`), a `TraceLog`,
  and an `ApprovalCard` that appears the moment a run pauses at
  `human_approval` — approving or rejecting calls `runStream.resume()`
  straight through the SSE-streaming API built earlier.
- **Use mode**: deliberately renders no graph at all — doc section 14 is
  explicit that "the graph is hidden" here — just the agent's goal, a
  Run button, an `ApprovalCard` if one comes up mid-run, and the final
  variables.
- **`ApprovalCard.tsx`** (new, in the canvas package so it's reusable
  beyond this app) renders doc section 13's exact worked example — "Agent
  wants to: / Refund ₹4,999 / Customer: XXXXX / Reason: Returned product /
  [Approve] [Reject]" — driven by `interrupt.context`, whatever fields the
  paused node happened to expose.
- **Self-bootstrapping for local dev**: on load, the app tries to fetch a
  demo agent by a fixed id; a 404 means the API is running against a fresh
  store, so it creates one from a bundled copy of `invoice-agent.json`.
  Zero manual setup to see something real on screen after `npm run dev`.

Verified with the strongest signal available without a browser: a genuine
`vite build` (208 modules — React Flow, every canvas package file, the
whole app) succeeds, and a real `vite` dev server was started and probed
with `curl`, confirming it actually serves the compiled app with HMR
wired up, not just that the files exist.

## Real-World Bug Fixes (from an actual local run)

Running this locally (Windows, Python 3.10, a browser hitting the app
shell) surfaced real bugs that every test suite up to this point had
missed — because they only exercise things this project's own tests
happened to cover, and a live browser session against a live server
exercises different failure modes (CORS, race conditions under real
network timing, cross-platform shell differences) than any of that. Each
of the following was independently verified against the actual code
before being fixed, not taken on faith:

- **Unbound `fetch`** (`apiClient.ts`) — `this.fetchImpl = options.fetchImpl
  ?? fetch` stored the bare function reference; calling it later as
  `this.fetchImpl(...)` invokes native `fetch` with the wrong `this`,
  which throws `Illegal invocation` in real browsers (Node's fetch is more
  forgiving, so no test here caught it). Fixed: bind to `globalThis`.
- **`typing.TypedDict` vs `typing_extensions.TypedDict`** (`state.py`) —
  works fine on the Python version this was built against, breaks under
  Pydantic 2.x on Python <3.12. Fixed, `typing-extensions` added as an
  explicit dependency.
- **No CORS middleware** — the app shell (`:5173`) and the API (`:8000`)
  are different origins; Swagger never caught this because it's served
  same-origin from the API itself. Added `CORSMiddleware`, configurable
  via `CORS_ALLOWED_ORIGINS`.
- **Bootstrap/session race** — `useAgentSession`'s own initial fetch ran
  independently of `App.tsx`'s demo-agent bootstrap effect; if bootstrap's
  create-agent call hadn't landed yet, the session's `GET` 404'd with
  nothing to retry it, so `spec` stayed `null` forever. Fixed with an
  `enabled` gate on the hook's initial fetch, wired to `bootstrapped`.
- **Real executors built but never wired into the API** — `compile_agent()`
  calls in every run/resume route omitted `executor_overrides` entirely,
  so the running app silently used stub AI/database executors regardless
  of what `real_executors/` could do. Fixed via a `get_executor_overrides`
  dependency that resolves to real Claude + SQLite when
  `ANTHROPIC_API_KEY` is set, with a clear one-time warning (not a silent
  fallback, not a crash) when it isn't.
- **Resume-stream trace duplication** — confirmed against my own earlier
  test output: a resumed run's checkpointed state already contains the
  full pre-pause trace, and `_stream_run`'s `seen_trace_len` started at 0
  for resumes too, re-emitting all prior events as if new. Fixed by
  seeding it from `graph.get_state()`; regression-locked with an exact
  assertion (`len(resume_trace) == 2`, not `>= 1`) so this can't silently
  regress again.
- **Non-atomic optimistic concurrency** — `apply_mutations()`'s version
  check happens against a Python-level snapshot; two concurrent requests
  could both pass it and both call the old blind `store.set()`, the
  second silently overwriting the first. Added real `compare_and_set` to
  both stores (a locked critical section for the in-memory store, a
  single `UPDATE ... WHERE version = expected` for Postgres) and proved
  the fix with an actual threaded race: 20 threads hitting the naive
  pattern under a realistic check-write gap produced 20 "winners" (19
  lost updates); the same race against `compare_and_set` produced exactly
  1, every time.
- **Canvas submitting mutations individually instead of as a batch** —
  `AgentCanvas` looped over `nodeChangesToMutations()`'s results calling
  `onMutation` once per mutation, defeating the mutation engine's atomic
  batching entirely (deleting 3 selected nodes could partially fail with
  spurious 409s). Fixed: `onMutation` now takes the whole array from one
  user action in a single call, threaded through `useAgentSession` and
  `AgentApiClient.applyMutations` (which already accepted a batch).
- **Chat silently dropping a successful response** — `sendInstruction`
  checked request staleness *before* appending the assistant's message to
  the transcript, so a reply that arrived after a concurrent canvas edit
  was discarded entirely — the user saw their question with no answer,
  even though the server had genuinely responded. Fixed: the message is
  always shown; only the resulting spec update is gated on staleness.
- **Approval buttons not disabled during resume** — `busy={false}` was
  hardcoded on one `ApprovalCard`, and simply omitted (defaulting to
  falsy) on the other. The direct fix (`busy={status === "running"}`)
  turned out to be nonsensical — TypeScript correctly flagged it as
  comparing two literal types that can't overlap within an already-
  narrowed `status === "paused"` block, since status can't be both values
  in one render. Replaced with an `ApprovalGate` wrapper that tracks its
  own "is this specific approval request in flight" state, independent of
  (and updated ahead of) the parent hook's `status`.
- **Misleading top-level error message** — any bootstrap failure, from a
  genuine network error to a 500 from a server that's very much reachable,
  rendered "Couldn't reach the API." Now distinguishes a network-level
  failure (`fetch` itself threw) from an `ApiError` (the server responded,
  just with an error status) and shows the right guidance for each.
- **`python3` hardcoded in the TS integration test** — breaks on Windows,
  where the executable is typically just `python`. Now resolves per-
  platform, overridable via `PYTHON_EXECUTABLE`.
- **README's `PYTHONPATH=.` syntax doesn't work in PowerShell** — added an
  explicit note with the PowerShell/cmd.exe equivalents.

**Not fixed in this pass, and why:** the report's remaining P2/P3 items
(auth/authorization, comprehensive `permissions`/`policies` enforcement,
automatic approval-policy insertion, real `loop`/`parallel`/`retry`/`wait`
semantics, real triggers, sub-agent execution, confidence-threshold
enforcement, `ai_decision` edges actually invoking a model rather than
`safe_eval`, a real agent list, and version-history UI) are legitimate
gaps, correctly identified in the original report as intentional prototype
limitations rather than accidental bugs, and each is a substantial enough
scope of its own to warrant a dedicated pass rather than being folded into
a bug-fix sweep. A published-version history store (report item #8) is
also deferred — `metadata.publishedVersion` currently records which
version *was* published, but the store doesn't retain a separate,
independently-fetchable copy of that spec once further edits move
`metadata.version` past it; fixing this properly means a real version
table, not a quick patch.

`backend/api/test_concurrency.py` is new — the only new automated test
this pass added beyond strengthening `test_streaming.py`'s existing
assertions, since most of these fixes (CORS, bootstrap race, chat
message-drop, approval busy state) are genuinely React/browser-level
behaviors that would need `@testing-library/react` + `jsdom` (not
currently a dependency) to test with the same rigor as this project's
existing pure-function suites; verified by careful code tracing instead
and flagged here rather than silently assumed correct.

## Chat Builder Fixed for Real, and Node Palette/Inspector Added

A live test against the running app surfaced that the chat builder was
completely broken — every instruction produced "Failed to fetch" in the
transcript with no real error visible. Root-caused with hard evidence
(not assumed): started the server with no `ANTHROPIC_API_KEY`, hit
`/instructions`, and inspected the **raw** response headers at the wire
level. Confirmed a real Starlette/FastAPI gotcha: a genuinely unhandled
exception (the Anthropic client failing with no auth configured) is
caught by `ServerErrorMiddleware`, which sits **outside** `CORSMiddleware`
no matter how `add_middleware()` is ordered — and even a registered
`@app.exception_handler(Exception)` doesn't fix it, since that also
attaches to the outer middleware, not the inner one `CORSMiddleware`
actually wraps. The resulting 500 had no `Access-Control-Allow-Origin`
header at all, which a cross-origin browser request can't read — it
surfaces in JS as a generic `TypeError: Failed to fetch`, with zero
indication anything even reached the server.

Fixed with a custom `@app.middleware("http")` that catches exceptions
itself and attaches the correct CORS header directly, rather than relying
on framework routing that turned out not to apply here. Verified twice: a
missing-header check before the fix (confirmed broken), then the same
check after (confirmed fixed) — plus a check that a **disallowed** origin
still gets no header, so the fix isn't an accidental open CORS bypass.
Locked in permanently as `backend/api/test_error_cors.py`.

Also closed two real feature gaps a live UI test surfaced that no
automated test could have caught, since they're about what's *possible to
do*, not whether existing code paths work correctly:

- **`nodeCatalog.ts` + `NodePalette.tsx`** — every `NodeType` the backend
  actually supports (all 29), categorized and described, draggable onto
  `AgentCanvas`. Previously the only way to get a new node onto the canvas
  at all was the NL compiler inferring one from a chat instruction — there
  was no direct, discoverable path. Dropping a palette entry computes the
  exact canvas position via `screenToFlowPosition` and applies the
  catalog entry's sensible defaults (e.g. `classify` gets a starter
  `config.labels` so it isn't immediately invalid) — the mapping itself
  (`catalogEntryToAddNodeMutation`) is a pure function, separated from the
  drag/drop DOM plumbing specifically so it's unit-testable without a
  browser.
- **`NodeInspector.tsx`** — clicking a node now surfaces its full details
  (type, category, execution mode, tool bindings, raw config as editable
  JSON) with Save and **Delete** actions. Node deletion technically
  already worked via React Flow's default Backspace-key handling (routed
  through the same `onNodesChange` → `nodeChangesToMutations` path
  documented in the mutation-batching section above) — but with zero
  visible affordance telling anyone that was possible. The inspector is
  now the direct, discoverable path for both.

`AgentCanvas` gained an `onSelectNode` prop (fired on node click / pane
click-to-deselect) so the app shell can drive the inspector, and `Build`
mode in `App.tsx` is now a three-pane layout: palette · canvas · (inspector
above chat). Verified with a real `vite build` (211 modules) after every
change in this section, plus 9 new catalog-consistency tests (checking,
among other things, that the palette's probabilistic-vs-deterministic
defaults actually match the compiler's own `PROBABILISTIC_CAPABLE_TYPES`
constraint) and 9 new tests for the drop-to-mutation mapping.

## V9 Fix-Spec: Tool Bindings, Structured Errors, AI-Unavailable UX

A comprehensive 65-section product spec (everything from "must fix now"
through full multi-tenant production features) named a specific,
concrete set of remaining bugs from the previous session's testing —
"Fix 4" through "Fix 9" in its own numbering. Rather than attempt the
full spec (an entire product roadmap, correctly triaged into P0/P1/P2 by
the doc itself), this pass addressed the verifiable, scoped items:

- **Fix 5 (the most consequential — a bug I'd just introduced)**:
  `NodePalette`, built last session, let a user drag ANY node type onto
  the canvas — including `search_web`, `send_email`, `database`, etc.,
  which the compiler requires a tool binding for. A freshly-dropped tool
  node validated fine structurally (Pydantic doesn't check this) but
  failed to COMPILE, with an error only ever discovered when the user
  tried to run it. Confirmed with a real reproduction before fixing:
  built the exact spec a drop would produce and fed it to
  `compile_agent()` directly. Fixed by having
  `catalogEntryToAddNodeMutation` auto-provision a placeholder
  `ToolBinding` with the exact capability the node type needs (mirroring
  the compiler's own `REQUIRED_CAPABILITY` mapping by hand, same
  cross-language pattern as elsewhere in this doc), submitted ATOMICALLY
  in the same mutation batch as the node — verified end-to-end by taking
  the exact JSON the frontend now generates and running it through the
  real Python mutation engine and compiler, confirming it compiles.
- **Fix 6**: `computeToolBindingWarnings` — a pure, client-side function
  (no round-trip needed) flags exactly this class of problem the moment
  it exists, rendered as a `⚠` badge directly on the node card.
- **Fix 8**: `CompileError` now carries a `node_id` (and error messages
  use the node's human label — `'Search Web'`, not `Node 'n_1'
  (search_web)`) end to end through `CompileCheckResponse` and every
  route that raises it. `App.tsx`'s new `CompileStatusBanner` proactively
  checks compilability whenever the spec's version changes and renders
  exactly the doc's own suggested format — "This agent can't run yet. ...
  [Configure Search Web]" — where the button jumps to Build mode with
  that node selected in the inspector (required lifting `selectedNodeId`
  state from `BuildMode` up to `App`, since Test mode needed to drive it
  too).
- **Fix 4**: a missing `ANTHROPIC_API_KEY` used to surface as a raw
  Anthropic SDK exception ("Could not resolve authentication method...")
  via the catch-all error middleware, reading as a platform bug rather
  than "you haven't configured this yet." Fixed with a proactive check —
  **specifically inside `get_anthropic_client()` itself**, not the route,
  after confirming a route-level check would incorrectly fire even when
  a test overrides the dependency with a fake client (dependency
  overrides replace a function's entire body, so a check placed inside
  it correctly never runs on the overridden path — verified with a
  dedicated regression test that a fake-client override still gets a
  clean 200, not the 503).

**A real regression caught and fixed during this same pass**: the Fix 4
change meant the missing-API-key scenario — previously the trigger
`test_error_cors.py` used to exercise a genuinely UNHANDLED exception —
now returns a clean, *handled* `HTTPException`, so that test's original
scenario stopped testing what it was built to test. The underlying
CORS-on-error-response fix is still necessary for any *future* unhandled
exception, so rather than delete the coverage, rewrote the test to inject
a fault via a dependency-overridden store whose `.get()` raises a raw
`RuntimeError` — exercising the real app and the real middleware stack
over genuine HTTP, without depending on any other part of the app
staying accidentally broken to serve as a test fixture.

All four fixes are also covered together in one place —
`backend/api/test_v9_fixes.py` — including the negative control that a
*different* 503 (not the no-API-key case) doesn't get misclassified by
the frontend's `ApiError.isAiUnavailable` getter.

## Suggested next build steps, in order

1. **Real Gmail/Sheets/search executors**: same pattern as
   `real_executors/database.py` — a function matching `NodeExecutor`'s
   signature, registered via `executor_overrides`. Needs real OAuth
   credentials this sandbox can't provide, so this is the natural next
   thing to build once there's a real deployment environment to test
   against.
2. **A real tool registry + credential management UI** (V9 spec sections
   8-10): the auto-provisioned placeholder bindings from Fix 5 are a
   stopgap — `toolId: "unconfigured.search_web"` is meant to be edited via
   `NodeInspector` into a real tool id, but there's no UI yet for
   *browsing* available tools or connecting real credentials
   (`credentialRef` exists in the schema, nothing populates it).
3. **Auth + multi-tenancy**: every route currently trusts `agent_id` alone;
   real deployment needs an owner check per doc section 19's permission
   model before this is safe to expose beyond localhost.
4. **A real agent list/creation flow**: the app shell currently
   self-bootstraps one hardcoded demo agent — the natural next UI is a
   picker over `GET /agents` (not yet a route — would need a
   `store.list()` method) plus a "describe what you want" creation form
   that seeds a brand-new spec via the NL compiler instead of a bundled
   JSON file.
5. **Publishing/versioning UI + a real version-history store**: the
   report correctly identified that `publishedVersion` currently records
   a number, not a retrievable spec — the store needs an actual versions
   table (or equivalent) before the app shell can expose a real Publish
   button, rollback, or "run the published version specifically" flow.
6. **Comprehensive permission/policy enforcement**: `permissions.allow`,
   `policies.toolAllowlist`, `dataAccessBoundaries`, `maxRunDurationSeconds`,
   `maxCostPerRun`, and automatic approval-policy insertion (so a
   `send_email` node with an `ask_permission` rule can't accidentally ship
   without an actual `human_approval` node) are currently spec/UI metadata
   more than runtime-enforced constraints — doc section 19's security
   model needs the compiler/runtime to actually check these, not just
   store them.

## Validating a spec

```bash
pip install -r backend/requirements.txt
python backend/validation/check_examples.py
```

## Running the demos

> **Windows/PowerShell:** the `PYTHONPATH=.` prefix below is Unix shell
> syntax and won't work directly in PowerShell. Use
> `$env:PYTHONPATH="."; python backend/run_examples.py` (note the `;`, not
> `&&`) instead, or set it once per session with `$env:PYTHONPATH = "."`
> and then run the plain `python ...` commands. cmd.exe uses
> `set PYTHONPATH=.` on its own line before the command. macOS/Linux users
> and WSL/Git Bash on Windows can use the commands exactly as written.

```bash
# Backend (Python)
pip install -r backend/requirements.txt
PYTHONPATH=. python backend/run_examples.py               # compiler: run + interrupt/resume
PYTHONPATH=. python backend/run_mutations.py              # mutation engine: batches + conflicts
PYTHONPATH=. python backend/run_nl_compiler.py            # NL compiler: instruction -> mutations (fake client)
PYTHONPATH=. python backend/run_real_executors.py         # real Claude structured outputs + real SQLite
PYTHONPATH=. python backend/api/test_app.py               # API layer: full lifecycle over real HTTP
PYTHONPATH=. python backend/persistence/test_postgres_persistence.py  # real Postgres durability, object-level (needs local Postgres, see above)
PYTHONPATH=. python backend/persistence/test_server_restart.py        # real Postgres durability, literal process restart (needs local Postgres)
PYTHONPATH=. python backend/api/test_streaming.py                     # SSE trace streaming: real incremental delivery, timing-verified
PYTHONPATH=. python backend/api/test_concurrency.py                   # atomic compare-and-set: real threaded race, proven closed

# Run the API server itself
PYTHONPATH=. uvicorn backend.api.app:app --reload
# then e.g.: curl -X POST localhost:8000/agents -H 'content-type: application/json' -d '{"spec": ...}'
# add DATABASE_URL=postgresql://agent_platform:agent_platform_dev@localhost/agent_platform
# to actually persist to Postgres instead of the in-memory default
# add ANTHROPIC_API_KEY=... to use real AI executors instead of stubs (see
# the Real Executors section above) — without it, the server logs a clear
# warning and falls back to stubs rather than failing every run

# Frontend (TypeScript)
npm install
npm run typecheck --workspaces
npm run test:canvas               # pure spec<->flow conversion tests
npm run test:canvas:integration   # spawns the real server, drives it with the real TS client

# App shell (needs the API server running — see above — at http://localhost:8000)
npm run dev     # http://localhost:5173 — Build/Test/Use modes, self-bootstraps a demo agent
npm run build   # production build, outputs to packages/app/dist
```

To go live with the NL compiler or the real AI executors, set
`ANTHROPIC_API_KEY` and construct `Anthropic()` without a `client=`
override (the API layer's `/instructions` route, and `build_ai_executors()`,
both do this automatically once nothing overrides the client).

To use the canvas in an app:
```tsx
import { AgentCanvas, AgentApiClient, useAgentSession } from "@agent-platform/canvas";

const client = new AgentApiClient({ baseUrl: "http://localhost:8000" });

function AgentEditor({ agentId }: { agentId: string }) {
  const { spec, applyMutation, resyncNonce, loading, error } = useAgentSession(client, agentId);
  if (loading || !spec) return <div>Loading…</div>;
  return (
    <>
      {error && <div>Edit failed: {error.message}</div>}
      <AgentCanvas spec={spec} onMutation={applyMutation} resyncNonce={resyncNonce} />
    </>
  );
}
```
