/** Minimal Markdown → safe HTML for Preview node (no external deps). */

function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

export function markdownToHtml(src: string): string {
  if (!src) return "<p style=\"color:#6b7280\">(empty)</p>";
  const lines = src.replace(/\r\n/g, "\n").split("\n");
  const out: string[] = [];
  let inCode = false;
  let codeBuf: string[] = [];
  let listType: "ul" | "ol" | null = null;

  const flushList = () => {
    if (listType) {
      out.push(listType === "ul" ? "</ul>" : "</ol>");
      listType = null;
    }
  };

  const inline = (text: string): string => {
    let t = escapeHtml(text);
    t = t.replace(/`([^`]+)`/g, "<code>$1</code>");
    t = t.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    t = t.replace(/\*([^*]+)\*/g, "<em>$1</em>");
    t = t.replace(/_([^_]+)_/g, "<em>$1</em>");
    return t;
  };

  for (const line of lines) {
    if (line.trim().startsWith("```")) {
      if (inCode) {
        out.push(`<pre style="background:#f3f4f6;padding:10px;border-radius:8px;overflow:auto;font-size:12px"><code>${escapeHtml(codeBuf.join("\n"))}</code></pre>`);
        codeBuf = [];
        inCode = false;
      } else {
        flushList();
        inCode = true;
      }
      continue;
    }
    if (inCode) {
      codeBuf.push(line);
      continue;
    }

    const h = /^(#{1,3})\s+(.+)$/.exec(line);
    if (h) {
      flushList();
      const level = h[1].length;
      out.push(`<h${level} style="margin:12px 0 6px;font-size:${level === 1 ? 18 : level === 2 ? 16 : 14}px">${inline(h[2])}</h${level}>`);
      continue;
    }

    const ul = /^[-*]\s+(.+)$/.exec(line);
    if (ul) {
      if (listType !== "ul") {
        flushList();
        out.push('<ul style="margin:6px 0;padding-left:1.25rem">');
        listType = "ul";
      }
      out.push(`<li>${inline(ul[1])}</li>`);
      continue;
    }

    const ol = /^(\d+)\.\s+(.+)$/.exec(line);
    if (ol) {
      if (listType !== "ol") {
        flushList();
        out.push('<ol style="margin:6px 0;padding-left:1.25rem">');
        listType = "ol";
      }
      out.push(`<li>${inline(ol[2])}</li>`);
      continue;
    }

    if (!line.trim()) {
      flushList();
      continue;
    }
    flushList();
    out.push(`<p style="margin:6px 0;line-height:1.5">${inline(line)}</p>`);
  }
  if (inCode) {
    out.push(`<pre style="background:#f3f4f6;padding:10px;border-radius:8px;overflow:auto;font-size:12px"><code>${escapeHtml(codeBuf.join("\n"))}</code></pre>`);
  }
  flushList();
  return out.join("\n") || "<p style=\"color:#6b7280\">(empty)</p>";
}
