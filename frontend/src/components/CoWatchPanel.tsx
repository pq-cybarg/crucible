import type { JSX } from "react";
import { useEffect, useRef, useState } from "react";
import { cowatchStream } from "../api";

// Watch a video WITH the AI: it streams commentary from a vision model while the clip plays, paced to
// real-time so the remarks track the timeline. Start your playback and the AI's live in the same moment.
function youtubeEmbed(url: string): string | null {
  const m = url.match(/(?:youtube\.com\/(?:watch\?v=|embed\/)|youtu\.be\/)([\w-]{11})/);
  return m ? `https://www.youtube.com/embed/${m[1]}` : null;
}

type Line = { kind: "commentary" | "reaction"; t: number; text: string };
const REACTION_ICON: Record<string, string> = { jumpscare: "😱", scene_cut: "⚡", loud: "🔊" };

export default function CoWatchPanel(): JSX.Element {
  const [source, setSource] = useState("");
  const [interval, setInterval] = useState(5);
  const [question, setQuestion] = useState("");
  const [lines, setLines] = useState<readonly Line[]>([]);
  const [watching, setWatching] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const feedRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => { feedRef.current?.scrollTo({ top: feedRef.current.scrollHeight }); }, [lines]);
  useEffect(() => () => abortRef.current?.abort(), []);

  const embed = youtubeEmbed(source);

  async function start(): Promise<void> {
    const src = source.trim();
    if (!src || watching) return;
    setLines([]); setErr(null); setWatching(true);
    const ac = new AbortController(); abortRef.current = ac;
    try {
      await cowatchStream(src, interval, question, (ev) => {
        if (ev.type === "commentary") {
          const d = ev.data as { t: number; text: string };
          if (d.text && d.text.trim()) setLines((p) => [...p, { kind: "commentary", t: d.t, text: d.text }]);
        } else if (ev.type === "reaction") {
          const d = ev.data as { t: number; type: string; intensity: number };
          const icon = REACTION_ICON[d.type] ?? "❗";
          setLines((p) => [...p, { kind: "reaction", t: d.t, text: `${icon} ${d.type.replace("_", " ")}` }]);
        } else if (ev.type === "done") setWatching(false);
      }, ac.signal);
    } catch (e: unknown) {
      if (!ac.signal.aborted) setErr(e instanceof Error ? e.message : "co-watch failed");
    } finally { setWatching(false); }
  }
  function stop(): void { abortRef.current?.abort(); setWatching(false); }

  return (
    <div className="panel">
      <div className="panel-head">
        <h1>co-<em>watch</em></h1>
        <p>Watch a video <b>with</b> the AI: it streams live commentary from your vision model as the clip
          plays, paced to the timeline. Paste a YouTube link (or a file path/URL), press <b>watch</b>, then
          hit play — the remarks land in sync. Needs a small vision model set in Preferences.</p>
      </div>

      <div className="cowatch-controls">
        <input className="in" placeholder="YouTube link or video URL/path…" value={source}
          onChange={(e) => setSource(e.target.value)} style={{ flex: "1 1 320px" }} />
        <label className="fld" style={{ flex: "0 0 auto" }}>every (s)
          <input className="in" type="number" min={1} max={30} value={interval}
            onChange={(e) => setInterval(Math.max(1, Number(e.target.value)))} style={{ width: 70 }} />
        </label>
        {watching
          ? <button className="btn" onClick={stop}>stop</button>
          : <button className="btn" disabled={source.trim().length === 0} onClick={() => void start()}>watch</button>}
      </div>
      <input className="in" placeholder="optional: what should it focus on / narrate?" value={question}
        onChange={(e) => setQuestion(e.target.value)} style={{ marginTop: 8, width: "100%" }} />
      {err && <div className="runtime-err" style={{ marginTop: 8 }}>{err}</div>}

      <div className="cowatch-body">
        <div className="cowatch-video">
          {embed
            ? <iframe title="cowatch" src={embed} allow="autoplay; encrypted-media" allowFullScreen />
            : <div className="hint">paste a YouTube link to embed the player here — or use any URL/path (the AI still watches; the player only shows for YouTube).</div>}
        </div>
        <div className="cowatch-feed" ref={feedRef}>
          <div className="engrave" style={{ margin: 0 }}>live commentary {watching && <span className="cowatch-live">● watching</span>}</div>
          {lines.length === 0 && !watching && <div className="hint">press watch to begin.</div>}
          {lines.map((l, i) => (
            <div key={i} className={`cowatch-line ${l.kind}`}><span className="cowatch-t">{l.t}s</span>{l.text}</div>
          ))}
        </div>
      </div>
    </div>
  );
}
