import type { JSX } from "react";

// Context explorer: a live, at-a-glance picture of what's in the model's window right now. Each turn
// is a segment sized by its (heuristic) token weight and coloured by role, laid out oldest→newest,
// with a marker where the compaction threshold sits. You SEE which turns are eating the context and
// what would be summarised on the next compaction, instead of guessing from a single number.
export type CtxTurn = { readonly role: string; readonly content: string };

function tokens(s: string): number {
  return Math.max(1, Math.floor((s ?? "").length / 4));
}

export default function ContextExplorer({ turns, limit, keepRecent = 6 }: {
  readonly turns: readonly CtxTurn[];
  readonly limit: number;
  readonly keepRecent?: number;
}): JSX.Element | null {
  if (turns.length === 0) return null;
  const weights = turns.map((t) => tokens(t.content));
  const total = weights.reduce((a, b) => a + b, 0);
  const scale = Math.max(total, limit);                       // so the limit marker is on-scale
  // the oldest turns beyond the last keepRecent are what compaction would summarise
  const oldCount = Math.max(0, turns.length - keepRecent);
  const markerPct = Math.min(100, (limit / scale) * 100);
  return (
    <div className="ctx-explorer" title={`≈ ${total} tokens across ${turns.length} turns (heuristic)`}>
      <div className="ctx-bar">
        {turns.map((t, i) => (
          <div key={i}
            className={`ctx-seg role-${t.role} ${i < oldCount ? "compactable" : "kept"}`}
            style={{ width: `${(weights[i]! / scale) * 100}%` }}
            title={`${t.role} · ≈${weights[i]} tok${i < oldCount ? " · would be summarised" : " · kept verbatim"}\n${t.content.slice(0, 140)}`} />
        ))}
        <div className="ctx-limit" style={{ left: `${markerPct}%` }} title={`compaction threshold ≈ ${limit} tokens`} />
      </div>
      <div className="ctx-legend">
        <span><i className="ctx-key role-user" /> you</span>
        <span><i className="ctx-key role-assistant" /> model</span>
        <span><i className="ctx-key compactable" /> would compact ({oldCount})</span>
        <span className="ctx-total">≈{total} / {limit} tok</span>
      </div>
    </div>
  );
}
