import type { ChatMessage } from "./api";

/** A console turn shaped just enough to project into the model's context window. */
export interface HistoryTurn {
  readonly kind: string;
  readonly text?: string;
}

/**
 * Project the console's turns into the MODEL CONTEXT WINDOW — ONLY the human↔model dialogue
 * (user + assistant messages).
 *
 * Everything else is deliberately excluded: tool-use, permission prompts, notices, memory tallies,
 * and — critically — ALL avatar ANIMATION state. The companion is animated by its own render loop
 * (ChatAvatar) and the server-side CompanionDriver, neither of which ever appears in `turns`; this
 * function is the enforced boundary so that rendering/animating her can never grow the context the
 * model sees ("animation context ≠ chat context"). Covered by chatHistory.test.ts.
 */
export function toModelHistory(turns: readonly HistoryTurn[]): readonly ChatMessage[] {
  return turns.flatMap((t): readonly ChatMessage[] =>
    (t.kind === "user" || t.kind === "assistant") && typeof t.text === "string"
      ? [{ role: t.kind, content: t.text }]
      : [],
  );
}
