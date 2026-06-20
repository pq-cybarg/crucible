import { useEffect, useMemo, useState } from "react";
import type { JSX } from "react";
import { motion } from "framer-motion";
import { getModels, getPublished, getSuite, runLmEval } from "../api";
import type { LmEvalResult, ModelRow, PublishedTable, SuiteTask } from "../api";

export default function BenchmarksPanel(): JSX.Element {
  const [models, setModels] = useState<readonly ModelRow[]>([]);
  const [suite, setSuite] = useState<readonly SuiteTask[]>([]);
  const [published, setPublished] = useState<PublishedTable>({});
  const [modelId, setModelId] = useState("");
  const [picked, setPicked] = useState<ReadonlySet<string>>(new Set(["gsm8k"]));
  const [limit, setLimit] = useState(25);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<LmEvalResult | null>(null);

  useEffect(() => {
    let alive = true;
    Promise.all([getModels(), getSuite(), getPublished()])
      .then(([m, s, p]) => {
        if (!alive) return;
        setModels(m);
        setSuite(s);
        setPublished(p);
        const live = m.find((row) => row.endpoint !== null);
        if (live) setModelId(live.id);
      })
      .catch(() => undefined);
    return () => {
      alive = false;
    };
  }, []);

  const liveModels = useMemo(() => models.filter((m) => m.endpoint !== null), [models]);
  const metrics = useMemo(() => {
    const out: string[] = [];
    for (const model of Object.values(published)) {
      for (const metric of Object.keys(model)) if (!out.includes(metric)) out.push(metric);
    }
    return out;
  }, [published]);
  const modelNames = useMemo(() => Object.keys(published), [published]);

  const toggle = (task: string): void => {
    const next = new Set(picked);
    if (next.has(task)) next.delete(task);
    else next.add(task);
    setPicked(next);
  };

  const run = async (): Promise<void> => {
    if (modelId === "" || picked.size === 0) return;
    setBusy(true);
    setResult(null);
    setResult(await runLmEval(modelId, [...picked], limit));
    setBusy(false);
  };

  const pct = (v: number): string => `${(v * 100).toFixed(1)}%`;

  return (
    <div className="panel">
      <div className="panel-head">
        <h1>benchmark <em>bay</em></h1>
        <p>Runs the EleutherAI lm-evaluation-harness — the canonical tool behind public leaderboards — against the local model. Real tasks, real per-metric standard error. Pick a suite and a sample limit; full runs scale to the 32B / GLM-5.2 nodes.</p>
      </div>

      <div className="abl-controls">
        <label className="fld">model
          <select className="in" value={modelId} onChange={(e) => setModelId(e.target.value)}>
            {liveModels.length === 0 && <option value="">— no live model (launch llama-server) —</option>}
            {liveModels.map((m) => <option key={m.id} value={m.id}>{m.name} · {m.quant}</option>)}
          </select>
        </label>
        <label className="fld">sample limit / task
          <input className="in" type="number" min={1} value={limit} onChange={(e) => setLimit(Number(e.target.value))} />
        </label>
        <button className="btn" onClick={() => void run()} disabled={busy || modelId === "" || picked.size === 0}>
          {busy ? "evaluating…" : `run ${picked.size} task${picked.size === 1 ? "" : "s"}`}
        </button>
      </div>

      <div className="engrave">canonical suite · lm-evaluation-harness</div>
      <div className="suite-grid">
        {suite.map((s) => (
          <button key={s.task} className={`suite-chip ${picked.has(s.task) ? "on" : ""}`} onClick={() => toggle(s.task)}>
            <span className="suite-name">{s.label}</span>
            <span className="suite-detail">{s.detail}</span>
            <code className="suite-task">{s.task}</code>
          </button>
        ))}
      </div>

      {result !== null && result.kind === "no-endpoint" && <div className="abl-note err">that model has no live endpoint — launch llama-server and register it.</div>}
      {result !== null && result.kind === "no-model" && <div className="abl-note err">model not found in registry.</div>}
      {result !== null && result.kind === "offline" && <div className="abl-note err">backend offline, or the run failed (check the model endpoint / task compatibility).</div>}
      {result !== null && result.kind === "results" && (
        <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}>
          <div className="engrave">measured · {modelId}</div>
          <table className="grid-table">
            <thead><tr><th>task</th><th>metric</th><th>filter</th><th>score</th><th>± stderr</th></tr></thead>
            <tbody>
              {result.rows.map((r, i) => (
                <tr key={`${r.task}-${r.metric}-${r.filter ?? ""}-${i}`}>
                  <td style={{ color: "var(--bone)" }}>{r.task}</td>
                  <td>{r.metric}</td>
                  <td style={{ color: "var(--ash)" }}>{r.filter ?? "—"}</td>
                  <td style={{ color: "var(--amber-bright)" }}>{pct(r.value)}</td>
                  <td style={{ color: "var(--ash)" }}>{r.stderr !== null ? `± ${(r.stderr * 100).toFixed(1)}%` : "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </motion.div>
      )}

      <div className="engrave">published frontier numbers · cited</div>
      <table className="grid-table">
        <thead><tr><th>metric</th>{modelNames.map((m) => <th key={m}>{m}</th>)}</tr></thead>
        <tbody>
          {metrics.map((metric) => (
            <tr key={metric}>
              <td style={{ color: "var(--bone)" }}>{metric}</td>
              {modelNames.map((name) => {
                const cell = published[name]?.[metric];
                return <td key={name}>{cell && cell.value !== null ? pct(cell.value) : <span style={{ color: "var(--ash)" }}>cite</span>}</td>;
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <p style={{ color: "var(--ash)", fontSize: 11, marginTop: 10 }}>
        Measured = run locally via lm-eval. Model columns = published/cited (GLM-5 family sourced; Opus left uncited rather than guessed). Note: chat-endpoint runs favour generative tasks (gsm8k, ifeval); loglikelihood MC tasks need the completions backend.
      </p>
    </div>
  );
}
