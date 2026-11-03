/**
 * Real cross-language, cross-process integration test: starts the actual
 * FastAPI server (backend/api/app.py, via uvicorn as a subprocess) and
 * drives it with the real AgentApiClient over genuine HTTP requests —
 * no TestClient, no mocks on either side. This is the strongest
 * verification available that the TS client and Python server actually
 * agree on the wire format (aliases, status codes, response shapes).
 *
 * Run:
 *   npx tsx packages/canvas/src/__tests__/apiClient.integration.test.ts
 * (starts and stops its own server; requires the backend/requirements.txt
 * deps to be installed and importable as `python3`.)
 */

import { spawn, type ChildProcess } from "node:child_process";
import { readFileSync } from "node:fs";
import { setTimeout as sleep } from "node:timers/promises";
import { AgentApiClient, ApiError } from "../apiClient";

const PORT = 8931;
const BASE_URL = `http://127.0.0.1:${PORT}`;
const REPO_ROOT = new URL("../../../../", import.meta.url).pathname;

let passed = 0;
let failed = 0;

function assert(condition: boolean, message: string) {
  if (condition) passed++;
  else {
    failed++;
    console.error(`FAIL: ${message}`);
  }
}

async function waitForServer(client: AgentApiClient, timeoutMs = 15_000): Promise<void> {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    try {
      await client.getAgent("__healthcheck_probe__");
      return; // any response (even 404) means the server is up
    } catch (e) {
      if (e instanceof ApiError) return; // server responded, just 404'd — that's fine, it's up
      await sleep(300);
    }
  }
  throw new Error(`Server did not become ready within ${timeoutMs}ms`);
}

/**
 * Windows typically only has a bare "python" on PATH, not "python3" (that
 * name is a Unix/macOS convention where "python" may be absent or point
 * to Python 2). PYTHON_EXECUTABLE lets CI or a developer override this
 * explicitly; otherwise this defaults per-platform rather than hardcoding
 * "python3" and silently failing to even spawn the server on Windows.
 */
function pythonExecutable(): string {
  return process.env.PYTHON_EXECUTABLE ?? (process.platform === "win32" ? "python" : "python3");
}

function startServer(): ChildProcess {
  const proc = spawn(
    pythonExecutable(),
    ["-m", "uvicorn", "backend.api.app:app", "--port", String(PORT), "--host", "127.0.0.1"],
    { cwd: REPO_ROOT, env: { ...process.env, PYTHONPATH: REPO_ROOT }, stdio: ["ignore", "pipe", "pipe"] }
  );
  proc.stderr?.on("data", () => {}); // uvicorn logs its startup banner to stderr; suppress noise
  return proc;
}

async function main() {
  console.log(`Starting FastAPI server on ${BASE_URL} ...`);
  const server = startServer();
  const client = new AgentApiClient({ baseUrl: BASE_URL });

  try {
    await waitForServer(client);
    console.log("Server is up.\n");

    // ---- create ----------------------------------------------------
    const specRaw = JSON.parse(readFileSync(`${REPO_ROOT}examples/research-agent.json`, "utf-8"));
    specRaw.metadata.id = "agt_ts_integration_" + Date.now();
    const created = await client.createAgent(specRaw);
    assert(created.metadata.id === specRaw.metadata.id, "createAgent: returns the created spec");
    console.log(`created ${created.metadata.id}, version ${created.metadata.version}`);

    // ---- get ----------------------------------------------------
    const fetched = await client.getAgent(created.metadata.id);
    assert(fetched.nodes.length === created.nodes.length, "getAgent: round-trips node count correctly");

    // ---- apply a real mutation using the TS AgentSpecMutation type --------
    const mutationResult = await client.applyMutations(
      created.metadata.id,
      [{ kind: "update_goal", goal: "Research a topic thoroughly and cite sources." }],
      created.metadata.version
    );
    assert(mutationResult.ok, "applyMutations: succeeds with a valid mutation + correct expected_version");
    assert(
      mutationResult.spec?.goal === "Research a topic thoroughly and cite sources.",
      "applyMutations: goal was actually updated"
    );
    const newVersion = mutationResult.spec!.metadata.version;
    assert(newVersion !== created.metadata.version, "applyMutations: version bumped");
    console.log(`mutation applied, version ${created.metadata.version} -> ${newVersion}`);

    // ---- stale version -> ApiError with isVersionConflict ----------------
    let conflictCaught = false;
    try {
      await client.applyMutations(
        created.metadata.id,
        [{ kind: "update_goal", goal: "stale" }],
        created.metadata.version // deliberately stale
      );
    } catch (e) {
      if (e instanceof ApiError && e.isVersionConflict) conflictCaught = true;
    }
    assert(conflictCaught, "applyMutations: stale expected_version surfaces as ApiError.isVersionConflict");
    console.log("stale-version conflict correctly surfaced through the real TS client");

    // ---- compile check over real HTTP ----------------------------------------------------
    const compileResult = await client.compileCheck(created.metadata.id);
    assert(compileResult.ok, "compileCheck: research agent compiles cleanly");

    // ---- run over real HTTP (no approval nodes in this spec, so it completes) ----------
    const runResult = await client.runAgent(created.metadata.id, { topic: "distributed consensus" });
    assert(runResult.status === "completed", `runAgent: expected completed, got ${runResult.status}`);
    assert(Array.isArray(runResult.trace) && runResult.trace.length > 0, "runAgent: trace was populated");
    console.log(`run completed, ${runResult.trace.length} trace events`);

    // ---- publish ----------------------------------------------------
    const publishResult = await client.publishAgent(created.metadata.id);
    assert(publishResult.publishedVersion === newVersion, "publishAgent: publishedVersion matches the version before publish");
    assert(publishResult.spec.metadata.status === "published", "publishAgent: status flipped to published");
    console.log(`published version ${publishResult.publishedVersion}`);

    console.log(`\n${passed} passed, ${failed} failed`);
    if (failed > 0) process.exitCode = 1;
  } finally {
    server.kill();
  }
}

main().catch((e) => {
  console.error(e);
  process.exitCode = 1;
});
