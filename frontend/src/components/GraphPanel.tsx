import type { JSX } from "react";
import { useMemo, useState } from "react";
import { motion } from "framer-motion";
import { runGraph } from "../api";
import type { GraphResult, GraphStage } from "../api";

// Model-graph builder: compose subsystems into a DAG and run it. Stage kinds:
//   model      — route a prompt to a model ({input} = merged upstream outputs)
//   transform  — pass merged inputs through
//   vote       — verifier ensemble: merge upstream outputs (majority / concat / first)
//   cascade    — cheap -> escalate: try each model in order until the output is accepted
//   tool       — invoke a registered tool
type Kind = "model" | "transform" | "vote" | "cascade" | "tool";
type Stage = {
  id: string;
  kind: Kind;
  inputs: string[];
  prompt: string;         // model / cascade
  modelId: string;        // model
  models: string;         // cascade: comma-separated ids ("" entry = default model)
  mustExclude: string;    // cascade: comma-separated phrases that force escalation
  strategy: string;       // vote
  toolName: string;       // tool
};

const KINDS: readonly Kind[] = ["model", "cascade", "vote", "transform", "tool"];

let _seq = 0;
function mkStage(kind: Kind = "model"): Stage {
  _seq += 1;
  return { id: `s${_seq}`, kind, inputs: [], prompt: "{input}", modelId: "", models: "",
    mustExclude: "i don't know, as an ai", strategy: "majority", toolName: "" };
}

// Build the backend stage config from the editable Stage.
function toStage(s: Stage): GraphStage {
  const config: Record<string, unknown> = {};
  if (s.kind === "model") {
    config.prompt = s.prompt;
    if (s.modelId.trim()) config.model_id = s.modelId.trim();
  } else if (s.kind === "cascade") {
    config.prompt = s.prompt;
    config.models = s.models.split(",").map((m) => (m.trim() ? m.trim() : null));
    const excl = s.mustExclude.split(",").map((x) => x.trim()).filter(Boolean);
    config.accept = { must_exclude: excl, min_len: 1 };
  } else if (s.kind === "vote") {
    config.strategy = s.strategy;
  } else if (s.kind === "tool") {
    config.name = s.toolName.trim();
  }
  return { id: s.id, kind: s.kind, inputs: s.inputs, config };
}

const PRESETS: Readonly<Record<string, () => Stage[]>> = {
  "verifier ensemble": () => {
    _seq = 0;
    const v1 = mkStage("model"), v2 = mkStage("model"), v3 = mkStage("model");
    v1.prompt = v2.prompt = v3.prompt = "Answer concisely: {input}";
    const merge = mkStage("vote");
    merge.inputs = [v1.id, v2.id, v3.id];
    merge.strategy = "majority";
    return [v1, v2, v3, merge];
  },
  "cascade (cheap→escalate)": () => {
    _seq = 0;
    const c = mkStage("cascade");
    c.prompt = "{input}";
    c.models = ", ";   // two default-model slots to demonstrate escalation
    return [c];
  },
  "pipeline": () => {
    _seq = 0;
    const a = mkStage("model"), b = mkStage("model");
    a.prompt = "Draft an answer to: {input}";
    b.prompt = "Improve and tighten this draft:\n{input}";
    b.inputs = [a.id];
    return [a, b];
  },
};

function renderOutput(value: unknown): JSX.Element {
  if (value !== null && typeof value === "object") {
    const o = value as Record<string, unknown>;
    // cascade dict
    if ("output" in o) {
      return (
        <div>
          <div className="graph-meta">chose <b>{String(o["chosen"])}</b>
            {o["escalated"] === true ? " · escalated" : ""} · {o["accepted"] === true ? "accepted" : "not accepted"}</div>
          <div className="graph-out-text">{String(o["output"])}</div>
        </div>
      );
    }
    // vote dict
    if ("result" in o) {
      return (
        <div>
          <div className="graph-meta">{String(o["strategy"])} of {String(o["n"])} · agreement {Math.round(Number(o["agreement"]) * 100)}%</div>
          <div className="graph-out-text">{String(o["result"])}</div>
        </div>
      );
    }
    return <pre className="graph-out-text">{JSON.stringify(o, null, 2)}</pre>;
  }
  return <div className="graph-out-text">{String(value ?? "")}</div>;
}

export default function GraphPanel(): JSX.Element {
  const [stages, setStages] = useState<Stage[]>(() => PRESETS["verifier ensemble"]());
  const [initial, setInitial] = useState("What is the capital of France?");
  const [result, setResult] = useState<GraphResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const ids = useMemo(() => stages.map((s) => s.id), [stages]);

  function patch(id: string, up: Partial<Stage>): void {
    setStages((prev) => prev.map((s) => (s.id === id ? { ...s, ...up } : s)));
  }
  function toggleInput(id: string, dep: string): void {
    setStages((prev) => prev.map((s) =>
      s.id === id ? { ...s, inputs: s.inputs.includes(dep) ? s.inputs.filter((d) => d !== dep) : [...s.inputs, dep] } : s));
  }
  function addStage(): void { setStages((prev) => [...prev, mkStage("model")]); }
  function removeStage(id: string): void {
    setStages((prev) => prev.filter((s) => s.id !== id).map((s) => ({ ...s, inputs: s.inputs.filter((d) => d !== id) })));
  }
  function loadPreset(name: string): void {
    setResult(null); setErr(null);
    setStages(PRESETS[name]?.() ?? []);
  }

  async function run(): Promise<void> {
    setBusy(true); setErr(null); setResult(null);
    try {
      setResult(await runGraph(stages.map(toStage), initial));
    } catch (e: unknown) {
      setErr(e instanceof Error ? e.message : "graph failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="panel">
      <div className="panel-head">
        <h1>model <em>graph</em></h1>
        <p>Compose subsystems into a DAG: chain models, fan out to a <b>verifier ensemble</b> (vote),
          or <b>cascade</b> cheap→strong (escalate only when the cheap answer hedges). Each stage's
          <code>{"{input}"}</code> is its upstream outputs, merged. Runs against the resolved model(s).</p>
      </div>

      <div className="graph-presets">
        <span className="engrave" style={{ margin: 0 }}>templates</span>
        {Object.keys(PRESETS).map((name) => (
          <button key={name} className="btn ghost" onClick={() => loadPreset(name)}>{name}</button>
        ))}
      </div>

      <label className="fld">initial input (feeds the root stages)
        <input className="in" value={initial} onChange={(e) => setInitial(e.target.value)} />
      </label>

      <div className="graph-stages">
        {stages.map((s) => (
          <div key={s.id} className="graph-stage">
            <div className="graph-stage-head">
              <code className="graph-id">{s.id}</code>
              <select className="byo-modelsel" value={s.kind}
                onChange={(e) => patch(s.id, { kind: e.target.value as Kind })}>
                {KINDS.map((k) => <option key={k} value={k}>{k}</option>)}
              </select>
              <button className="graph-del" title="remove stage" onClick={() => removeStage(s.id)}>✕</button>
            </div>

            <div className="graph-inputs">
              <span className="graph-lbl">inputs</span>
              {ids.filter((i) => i !== s.id).length === 0 && <span className="graph-hint">root (uses initial)</span>}
              {ids.filter((i) => i !== s.id).map((dep) => (
                <label key={dep} className="graph-chip">
                  <input type="checkbox" checked={s.inputs.includes(dep)} onChange={() => toggleInput(s.id, dep)} />
                  {dep}
                </label>
              ))}
            </div>

            {(s.kind === "model" || s.kind === "cascade") && (
              <textarea className="graph-prompt" value={s.prompt} rows={2}
                onChange={(e) => patch(s.id, { prompt: e.target.value })} placeholder="prompt · {input} = upstream" />
            )}
            {s.kind === "model" && (
              <input className="in" value={s.modelId} placeholder="model id (blank = default)"
                onChange={(e) => patch(s.id, { modelId: e.target.value })} />
            )}
            {s.kind === "cascade" && (
              <div className="graph-cfg">
                <input className="in" value={s.models} placeholder="models, cheap→strong (comma; blank = default)"
                  onChange={(e) => patch(s.id, { models: e.target.value })} />
                <input className="in" value={s.mustExclude} placeholder="escalate if output contains (comma)"
                  onChange={(e) => patch(s.id, { mustExclude: e.target.value })} />
              </div>
            )}
            {s.kind === "vote" && (
              <select className="byo-modelsel" value={s.strategy} onChange={(e) => patch(s.id, { strategy: e.target.value })}>
                <option value="majority">majority (+ agreement)</option>
                <option value="concat">concat</option>
                <option value="first">first non-empty</option>
              </select>
            )}
            {s.kind === "tool" && (
              <input className="in" value={s.toolName} placeholder="tool name"
                onChange={(e) => patch(s.id, { toolName: e.target.value })} />
            )}
          </div>
        ))}
      </div>

      <div className="graph-actions">
        <button className="btn ghost" onClick={addStage}>+ stage</button>
        <button className="btn" onClick={() => void run()} disabled={busy || stages.length === 0}>
          {busy ? "running…" : "run graph"}
        </button>
        {err && <span className="runtime-err">{err}</span>}
      </div>

      {result && (
        <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}>
          <div className="engrave">order · {result.order.join(" → ")}</div>
          {result.order.map((id) => (
            <div key={id} className={`graph-result ${id in result.result ? "terminal" : ""}`}>
              <code className="graph-id">{id}{id in result.result ? " · output" : ""}</code>
              {renderOutput(result.outputs[id])}
            </div>
          ))}
        </motion.div>
      )}
    </div>
  );
}
