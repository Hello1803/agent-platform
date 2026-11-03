/**
 * A fetch-based SSE reader, used instead of the browser's EventSource
 * because EventSource only supports GET requests with no custom body —
 * our streaming endpoints (POST /agents/{id}/run/stream) need a JSON body
 * (seed_variables). This is a common, well-established pattern for SSE
 * over POST; nothing exotic about it.
 *
 * The parsing logic (`parseSSEChunk`) is deliberately a pure function
 * separate from the fetch/stream plumbing, so it's unit-testable without
 * a real network stream — see __tests__/sseClient.test.ts.
 */

export interface SSEEvent {
  event: string;
  data: unknown;
}

/**
 * Parses a buffer of raw SSE text into complete events plus any leftover
 * partial event (a chunk boundary can land mid-event). Frames look like:
 *   event: trace\ndata: {"node_id": "n1", ...}\n\n
 * Returns the events found and the remainder to prepend to the next chunk.
 */
export function parseSSEChunk(buffer: string): { events: SSEEvent[]; remainder: string } {
  const events: SSEEvent[] = [];
  const frames = buffer.split("\n\n");
  // The last element is either an empty string (buffer ended exactly on a
  // frame boundary) or a partial frame to carry over to the next chunk.
  const remainder = frames.pop() ?? "";

  for (const frame of frames) {
    if (!frame.trim()) continue;
    let eventName = "message";
    let dataLine: string | null = null;
    for (const line of frame.split("\n")) {
      if (line.startsWith("event:")) eventName = line.slice("event:".length).trim();
      else if (line.startsWith("data:")) dataLine = line.slice("data:".length).trim();
    }
    if (dataLine !== null) {
      try {
        events.push({ event: eventName, data: JSON.parse(dataLine) });
      } catch {
        // Malformed data line — skip rather than throw, so one bad frame
        // doesn't take down the whole stream for the caller.
      }
    }
  }

  return { events, remainder };
}

export interface StreamSSEOptions {
  url: string;
  body: unknown;
  onEvent: (event: SSEEvent) => void;
  fetchImpl?: typeof fetch;
  signal?: AbortSignal;
}

/** Streams SSE frames from a POST endpoint, calling onEvent for each parsed frame as it arrives. Resolves when the server closes the stream. */
export async function streamSSE({ url, body, onEvent, fetchImpl, signal }: StreamSSEOptions): Promise<void> {
  const doFetch = fetchImpl ?? fetch;
  const response = await doFetch(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });

  if (!response.ok || !response.body) {
    const text = await response.text().catch(() => "");
    throw new Error(`SSE request failed: ${response.status} ${text}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const { events, remainder } = parseSSEChunk(buffer);
    buffer = remainder;
    for (const event of events) onEvent(event);
  }
}
