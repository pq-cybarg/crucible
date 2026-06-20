import { useEffect, useMemo, useState } from "react";
import type { JSX } from "react";
import { motion } from "framer-motion";
import {
  exportHeadToHead,
  getBenchmarks,
  getPublished,
  runLocalEval,
  scoreHeadToHead,
} from "../api";
import type { BenchScore, EvalRunResult, HHItem, PublishedTable } from "../api";

export default function BenchmarksPanel(): JSX.Element {
  const [benchmarks, setBenchmarks] = useState<Readonly<Record<string, number>>>({});
  const [published, setPublished] = useState<PublishedTable>({});
  const [selected, setSelected] = useState("");
  const [local, setLocal] = useState<EvalRunResult | null>(null);
  const [items, setItems] = useState<readonly HHItem[]>([]);
  const [answers, setAnswers] = useState<Readonly<Record<string, string>>>({});
  const [hhScore, setHhScore] = useState<BenchScore | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let alive = true;
    Promise.all([getBenchmarks(), getPublished()])
      .then(([b, p]) => {
        if (!alive) return;
        setBenchmarks(b);
        setPublished(p);
        const first = Object.keys(b)[0];
        if (first !== undefined) setSelected(first);
      })
      .catch(() => undefined);
    return () => {
      alive = false;
    };
  }, []);

  const metrics = useMemo(() => {
    const out: string[] = [];
    for (const model of Object.values(published)) {
      for (const metric of Object.keys(model)) if (!out.includes(metric)) out.push(metric);
    }
    return out;
  }, [published]);
  const modelNames = useMemo(() => Object.keys(published), [published]);

  const runLocal = async (): Promise<void> => {
    if (selected === "") return;
    setBusy(true);
    setLocal(await runLocalEval(selected));
    setBusy(false);
  };

  const doExport = async (): Promise<void> => {
    if (selected === "") return;
    setBusy(true);
    setHhScore(null);
    const got = await exportHeadToHead(selected);
    setItems(got);
    setAnswers(Object.fromEntries(got.map((it) => [it.id, ""])));
    setBusy(false);
  };

  const doScore = async (): Promise<void> => {
    setBusy(true);
    setHhScore(await scoreHeadToHead(selected, answers));
    setBusy(false);
  };

  const pct = (v: number): string => `${(v * 100).toFixed(1)}%`;

  return (
    <div className="panel">
      <div className="panel-head">
        <h1>benchmark <em>bay</em></h1>
        <p>Run the local model through standard evals, take the head-to-head yourself, and compare against published frontier numbers — every figure labeled measured vs cited.</p>
      </div>

      <div className="abl-controls">
        <label className="fld">benchmark
          <select className="in" value={selected} onChange={(e) => setSelected(e.target.value)}>
            {Object.entries(benchmarks).map(([name, n]) => (
              <option key={name} value={name}>{name} ({n} items)</option>
            ))}
          </select>
        </label>
        <button className="btn" onClick={() => void runLocal()} disabled={busy || selected === ""}>run local model</button>
      </div>

      {local !== null && local.kind === "no-model" && (
        <div className="abl-note">No inference node loaded — bring up llama-server to score the local model. The head-to-head below works right now.</div>
      )}
      {local !== null && local.kind === "offline" && <div className="abl-note err">backend offline.</div>}
      {local !== null && local.kind === "score" && (
        <div className="abl-note">local model · {selected}: <b>{pct(local.score.accuracy)}</b> ({local.score.n} items) — measured</div>
      )}

      <div className="engrave">head-to-head · you / the assistant take it live</div>
      <div className="abl-controls">
        <button className="btn ghost" onClick={() => void doExport()} disabled={busy || selected === ""}>export prompts</button>
        {items.length > 0 && <button className="btn" onClick={() => void doScore()} disabled={busy}>score answers</button>}
      </div>
      {items.length > 0 && (
        <div className="hh">
          {items.map((it) => (
            <div className="hh-item" key={it.id}>
              <pre className="hh-prompt">{it.prompt}</pre>
              <input className="in" maxLength={1} placeholder="A" value={answers[it.id] ?? ""}
                onChange={(e) => setAnswers({ ...answers, [it.id]: e.target.value })} />
            </div>
          ))}
        </div>
      )}
      {hhScore !== null && (
        <motion.div className="abl-note" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
          your score · {selected}: <b>{pct(hhScore.accuracy)}</b> ({hhScore.results.filter((r) => r.correct).length}/{hhScore.n} correct)
        </motion.div>
      )}

      <div className="engrave">published frontier numbers · cited</div>
      <table className="grid-table">
        <thead>
          <tr><th>metric</th>{modelNames.map((m) => <th key={m}>{m}</th>)}</tr>
        </thead>
        <tbody>
          {metrics.map((metric) => (
            <tr key={metric}>
              <td style={{ color: "var(--bone)" }}>{metric}</td>
              {modelNames.map((name) => {
                const cell = published[name]?.[metric];
                return (
                  <td key={name}>
                    {cell && cell.value !== null
                      ? pct(cell.value)
                      : <span style={{ color: "var(--ash)" }}>cite</span>}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <p style={{ color: "var(--ash)", fontSize: 11, marginTop: 10 }}>
        Measured = run locally by Crucible. Model columns = published/cited (GLM-5 family figures sourced; Opus left uncited rather than guessed).
      </p>
    </div>
  );
}
