import type { JSX } from "react";
import { useEffect, useRef } from "react";

export interface ToolFeedItem {
  readonly id: string;
  readonly name: string;
  readonly args: unknown;
  readonly status: string;          // "running" | "ok" | "fail"
  readonly output: string;
}

/**
 * A compact activity log of the agent's TOOL calls, docked UNDER the companion so tool-use no longer
 * floods the conversation thread (the chat stays the human↔model dialogue). Newest at the bottom,
 * auto-scrolled; a status dot per row (running = amber pulse, ok = green, fail = red). Hover a row for
 * its args; failures show their output inline. Renders nothing until the first tool runs.
 */
export default function ToolFeed({ items }: { readonly items: readonly ToolFeedItem[] }): JSX.Element | null {
  const listRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    const el = listRef.current;
    if (el) el.scrollTop = el.scrollHeight;      // keep the latest call in view
  }, [items.length, items[items.length - 1]?.status]);

  if (items.length === 0) return null;
  const recent = items.slice(-40);               // cap the DOM; the thread keeps the full history via context
  return (
    <div className="toolfeed" aria-label="tool activity">
      <div className="toolfeed-head">tools · {items.length}</div>
      <div className="toolfeed-list" ref={listRef}>
        {recent.map((it) => (
          <div key={it.id} className={`toolfeed-row ${it.status}`} title={safeArgs(it.args)}>
            <span className="tf-dot" aria-hidden="true" />
            <span className="tf-name">{it.name}</span>
            {it.status === "fail" && it.output.length > 0 && (
              <span className="tf-err" title={it.output}>failed</span>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

function safeArgs(args: unknown): string {
  try {
    const s = JSON.stringify(args);
    return s && s.length > 300 ? `${s.slice(0, 300)}…` : (s ?? "");
  } catch {
    return "";
  }
}
