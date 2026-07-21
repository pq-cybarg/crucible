// Client-side, CONTEXT-FREE sentiment → companion mood. Scans the assistant's VISIBLE reply text for
// emotional cues and returns a mood + intensity to flash on the in-chat avatar for a beat (heart/star eyes
// on strong emotional moments, softer moods otherwise). Nothing here is sent to the model — this reads text
// already on screen, so animating her from it never touches the context window (#31). Heuristic on purpose:
// conservative, only the special "effect" eyes (lovestruck/starstruck/dizzy) fire, and only on clear cues.

export interface MoodHit {
  readonly mood: string;
  readonly weight: number;   // ≥0.55 crosses the backend's shape-intensity gate; lower = a soft normal mood
}

export function moodFromText(text: string): MoodHit | null {
  const t = text.toLowerCase();
  const exclaims = (text.match(/!/g) ?? []).length;
  const has = (re: RegExp): boolean => re.test(t);

  // strong affection → heart eyes (the rare "particularly emotional" beat)
  if (has(/❤️|💕|💖|😍|🥰/) || (has(/\blove(d|s|ly)?\b|adorable|sweetheart|precious/) && exclaims >= 1)) {
    return { mood: "lovestruck", weight: 0.9 };
  }
  // admiration → sparkle/kirakira eyes
  if (has(/gorgeous|beautiful|stunning|magnificent|so pretty|perfect!|flawless/)) return { mood: "sparkly", weight: 0.85 };
  // wonder / hype → star eyes
  if (has(/⭐|🌟|✨|🤩/) || has(/incredible|amazing|awesome|fantastic|brilliant|wow\b/) || exclaims >= 3) {
    return { mood: "starstruck", weight: 0.85 };
  }
  // mirth → laughing (the ^ squint)
  if (has(/😂|🤣|\bhaha+\b|\blol\b|\blmao\b|hilarious/)) return { mood: "laughing", weight: 0.8 };
  // shock → pinprick/dot eyes
  if (has(/😱|😳|oh no|no way|wait,? what/) || /\?!|!\?/.test(text)) return { mood: "shock", weight: 0.8 };
  // grief → crying/tears (before the softer "sad")
  if (has(/😭|😢|heartbreak|devastat|so sad|makes me (want to )?cry/)) return { mood: "crying", weight: 0.75 };
  // confusion → swirl eyes
  if (has(/🥴|😵|confus|baffl|no idea|not sure what|can'?t tell/)) return { mood: "dizzy", weight: 0.75 };
  // ko / dead-tired → X eyes
  if (has(/💀|i'?m dead|i can'?t even|so done|exhaust/)) return { mood: "ko", weight: 0.8 };
  // apology / failure → sad (soft, no special eyes)
  if (has(/😔|\bsorry\b|unfortunately|apologi|\berror\b|\bfailed\b|couldn'?t/)) return { mood: "sad", weight: 0.55 };
  // curiosity → curious (soft)
  if (text.trim().endsWith("?") || has(/curious|interesting|i wonder|good question/)) return { mood: "curious", weight: 0.5 };
  // mild positive → happy (soft)
  if (exclaims >= 1 || has(/\bglad\b|\bgreat\b|\bnice\b|\bhappy\b|\bthanks?\b|done!?$/)) return { mood: "happy", weight: 0.55 };
  return null;
}
