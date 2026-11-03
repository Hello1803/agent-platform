import { parseSSEChunk } from "../sseClient";

let passed = 0;
let failed = 0;

function assert(condition: boolean, message: string) {
  if (condition) passed++;
  else {
    failed++;
    console.error(`FAIL: ${message}`);
  }
}

function assertEqual<T>(actual: T, expected: T, message: string) {
  const ok = JSON.stringify(actual) === JSON.stringify(expected);
  assert(ok, `${message} — expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`);
}

// ---- single complete event ----

{
  const { events, remainder } = parseSSEChunk('event: trace\ndata: {"node_id":"n1"}\n\n');
  assertEqual(events, [{ event: "trace", data: { node_id: "n1" } }], "parses a single complete event");
  assertEqual(remainder, "", "no remainder when buffer ends exactly on a frame boundary");
}

// ---- multiple events in one chunk ----

{
  const buf = 'event: trace\ndata: {"a":1}\n\nevent: trace\ndata: {"a":2}\n\nevent: done\ndata: {"status":"completed"}\n\n';
  const { events, remainder } = parseSSEChunk(buf);
  assertEqual(events.length, 3, "parses multiple events from one chunk");
  assertEqual(events[2], { event: "done", data: { status: "completed" } }, "last event parsed correctly");
  assertEqual(remainder, "", "no remainder");
}

// ---- THE important case: an event split across two chunks ----

{
  // Simulates what a real TCP/HTTP chunk boundary can do: cut a frame in half.
  const chunk1 = 'event: trace\ndata: {"node_id":"n1","detail":"executed clas';
  const { events: events1, remainder: remainder1 } = parseSSEChunk(chunk1);
  assertEqual(events1, [], "no complete event yet — the frame was cut mid-data");
  assertEqual(remainder1, chunk1, "the entire partial chunk is carried forward as remainder");

  const chunk2 = 'sify"}\n\nevent: done\ndata: {"status":"paused"}\n\n';
  const { events: events2 } = parseSSEChunk(remainder1 + chunk2);
  assertEqual(events2.length, 2, "the reassembled frame plus the next one both parse correctly");
  assertEqual(
    events2[0],
    { event: "trace", data: { node_id: "n1", detail: "executed classify" } },
    "the split event's data reassembled correctly across the chunk boundary"
  );
  assertEqual(events2[1], { event: "done", data: { status: "paused" } }, "the following event also parsed");
}

// ---- malformed data line is skipped, not thrown ----

{
  const buf = 'event: trace\ndata: {not valid json\n\nevent: trace\ndata: {"ok":true}\n\n';
  const { events } = parseSSEChunk(buf);
  assertEqual(events, [{ event: "trace", data: { ok: true } }], "malformed frame skipped, valid one still parsed");
}

// ---- default event name when 'event:' line is absent ----

{
  const { events } = parseSSEChunk('data: {"x":1}\n\n');
  assertEqual(events, [{ event: "message", data: { x: 1 } }], "defaults to 'message' when no event: line given");
}

console.log(`\n${passed} passed, ${failed} failed`);
if (failed > 0) process.exit(1);
