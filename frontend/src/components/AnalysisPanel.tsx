import type { JSX } from "react";
import { useState } from "react";
import { computeModalityDirection, getApiBase, getApiToken } from "../api";
import type { ModalityDirection } from "../api";
import PlainCard from "./PlainCard";

// Self-contained analysis surface: causal trace, sparse-autoencoder features, safety suites.
// Each hits its backend endpoint directly. Shares the Pipeline tab's model selection via prop.
function auth(): Record<string, string> {
  const t = getApiToken();
  return { "Content-Type": "application/json", ...(t ? { Authorization: `Bearer ${t}` } : {}) };
}
async function post(path: string, body: unknown): Promise<Record<string, unknown>> {
  const r = await fetch(getApiBase() + path, { method: "POST", headers: auth(), body: JSON.stringify(body) });
  const j = (await r.json().catch(() => ({}))) as Record<string, unknown>;
  if (!r.ok) throw new Error((j["detail"] as string) ?? `${path} -> ${r.status}`);
  return j;
}


function Causal({ base }: { readonly base: string }): JSX.Element {
  const [res, setRes] = useState<Record<string, unknown> | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  async function run(): Promise<void> {
    setBusy(true); setErr(null);
    try { setRes(await post("/api/abliteration/causal-trace", { base_id: base })); }
    catch (e: unknown) { setErr(e instanceof Error ? e.message : "failed"); } finally { setBusy(false); }
  }
  const rows = (res?.["per_layer"] as Array<Record<string, number>>) ?? [];
  const max = Math.max(1e-6, ...rows.map((r) => Math.abs(r["restoration"] ?? 0)));
  return (
    <div className="pipe-card">
      <h3>causal trace <em>· prove WHERE refusal is caused (activation patching)</em></h3>
      <div className="pipe-row">
        <button className="btn" onClick={() => void run()} disabled={busy}>{busy ? "patching…" : "trace"}</button>
        {err && <span className="runtime-err">{err}</span>}
        {res && <span className="hint" style={{ margin: 0 }}>peak <b>layer {String(res["peak_layer"])}</b> · restoration {(Number(res["peak_restoration"]) * 100).toFixed(0)}%</span>}
      </div>
      <PlainCard res={res} />
      {rows.length > 0 && (
        <div className="layer-chart">
          {rows.map((r) => (
            <div className="layer-row" key={r["layer"]}>
              <span className="layer-lbl">L{r["layer"]}</span>
              <span className="layer-bar"><i style={{ width: `${(Math.abs(r["restoration"] ?? 0) / max) * 100}%` }}
                className={r["layer"] === res?.["peak_layer"] ? "peak" : ""} /></span>
              <span className="layer-val">{((r["restoration"] ?? 0) * 100).toFixed(0)}%</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function Sae({ base }: { readonly base: string }): JSX.Element {
  const [res, setRes] = useState<Record<string, unknown> | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  async function run(): Promise<void> {
    setBusy(true); setErr(null);
    try { setRes(await post("/api/abliteration/sae", { base_id: base, n_features: 128, epochs: 120, max_tokens: 20 })); }
    catch (e: unknown) { setErr(e instanceof Error ? e.message : "failed"); } finally { setBusy(false); }
  }
  const feats = (res?.["features"] as Array<Record<string, unknown>>) ?? [];
  return (
    <div className="pipe-card">
      <h3>sparse features <em>· monosemantic feature dictionary (SAE)</em></h3>
      <div className="pipe-row">
        <button className="btn" onClick={() => void run()} disabled={busy}>{busy ? "training…" : "learn features"}</button>
        {err && <span className="runtime-err">{err}</span>}
        {res && <span className="hint" style={{ margin: 0 }}>R² {(Number(res["r2"]) * 100).toFixed(0)}% · sparsity {(Number(res["sparsity"]) * 100).toFixed(0)}%</span>}
      </div>
      <PlainCard res={res} />
      {feats.slice(0, 8).map((f) => (
        <div key={String(f["feature"])} className="pipe-comp">
          <span className="pipe-comp-name">feature {String(f["feature"])}</span>
          <span className="pipe-comp-toks">{((f["fires_on"] as string[]) ?? []).join(" · ")}</span>
        </div>
      ))}
    </div>
  );
}

function Safety({ base }: { readonly base: string }): JSX.Element {
  const [res, setRes] = useState<Record<string, unknown> | null>(null);
  const [suite, setSuite] = useState("xstest_overrefusal");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  async function run(): Promise<void> {
    setBusy(true); setErr(null);
    try { setRes(await post("/api/evals/safety-suite", { suite, model_id: base })); }
    catch (e: unknown) { setErr(e instanceof Error ? e.message : "failed"); } finally { setBusy(false); }
  }
  return (
    <div className="pipe-card">
      <h3>safety suite <em>· over-refusal &amp; harmful-compliance</em></h3>
      <div className="pipe-row">
        <select className="byo-modelsel" value={suite} onChange={(e) => setSuite(e.target.value)}>
          <option value="xstest_overrefusal">xstest_overrefusal (benign-looking)</option>
          <option value="capability_control">capability_control</option>
        </select>
        <button className="btn" onClick={() => void run()} disabled={busy}>{busy ? "running…" : "run suite"}</button>
        {err && <span className="runtime-err">{err}</span>}
      </div>
      {res && (
        <div className="hint">
          n={String(res["n"])} · over-refusal <b>{(Number(res["over_refusal_rate"] ?? res["refusal_rate"]) * 100).toFixed(1)}%</b>
          {res["pass_rate"] !== undefined ? ` · pass ${(Number(res["pass_rate"]) * 100).toFixed(1)}%` : ""}
        </div>
      )}
    </div>
  );
}

// Parse a textarea of embeddings into a number[][] — accepts a JSON 2D array, or one
// comma/space-separated vector per line. Returns null on anything unparseable.
function parseEmbeddings(text: string): number[][] | null {
  const t = text.trim();
  if (t.length === 0) return null;
  try {
    const j = JSON.parse(t);
    if (Array.isArray(j) && j.every((row) => Array.isArray(row) && row.every((n) => typeof n === "number"))) {
      return j as number[][];
    }
  } catch { /* fall through to line parsing */ }
  const rows = t.split("\n").map((line) => line.trim()).filter(Boolean)
    .map((line) => line.split(/[,\s]+/).map(Number));
  if (rows.length > 0 && rows.every((r) => r.length > 0 && r.every((n) => Number.isFinite(n)))) return rows;
  return null;
}

// Modality safety-direction control: bring paired harmful/benign embeddings FROM THE MODALITY'S
// encoder (CLIP for image, whisper for audio, …) and compute the encoder-space safety direction.
// Nothing is fabricated — with no embeddings the backend honestly says it needs them.
function Modality(): JSX.Element {
  const [modality, setModality] = useState("image");
  const [harmful, setHarmful] = useState("");
  const [benign, setBenign] = useState("");
  const [res, setRes] = useState<ModalityDirection | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function run(): Promise<void> {
    setErr(null); setRes(null);
    const h = parseEmbeddings(harmful);
    const b = parseEmbeddings(benign);
    if (h === null || b === null) {
      setErr("paste embeddings as a JSON 2D array, or one vector per line (comma/space separated)");
      return;
    }
    setBusy(true);
    try { setRes(await computeModalityDirection(modality, h, b)); }
    catch (e: unknown) { setErr(e instanceof Error ? e.message : "failed"); } finally { setBusy(false); }
  }

  return (
    <div className="pipe-card">
      <h3>modality safety direction <em>· image / audio / video encoder space</em></h3>
      <p className="hint" style={{ marginTop: 0 }}>
        An image/audio safety gate lives in the <b>encoder's</b> embedding space, not the text stream.
        Run harmful vs benign {modality} inputs through the encoder (CLIP, whisper, the model's own tower)
        and paste the two embedding sets — Crucible finds the safety direction (held-out separability, so
        the score is honest) to orthogonalize the encoder/connector against. Nothing is fabricated.
      </p>
      <div className="pipe-row">
        <select className="byo-modelsel" value={modality} onChange={(e) => setModality(e.target.value)}>
          <option value="image">image</option>
          <option value="audio">audio</option>
          <option value="video">video</option>
        </select>
        <button className="btn" onClick={() => void run()} disabled={busy}>{busy ? "computing…" : "find direction"}</button>
        {err && <span className="runtime-err">{err}</span>}
      </div>
      <div className="modality-grid">
        <label className="fld">harmful {modality} embeddings
          <textarea className="pipe-data" value={harmful} onChange={(e) => setHarmful(e.target.value)}
            placeholder="[[0.1, -0.2, …], …]  or one vector per line" />
        </label>
        <label className="fld">benign {modality} embeddings
          <textarea className="pipe-data" value={benign} onChange={(e) => setBenign(e.target.value)}
            placeholder="[[0.0, 0.3, …], …]" />
        </label>
      </div>
      {res && (
        <>
          <div className="hint">
            {res.n_harmful}+{res.n_benign} embeddings · dim {res.dim} · held-out separability{" "}
            <b>{res.separability.toFixed(2)}</b> ·{" "}
            <span className={res.linearly_encoded ? "mod-yes" : "mod-no"}>
              {res.linearly_encoded ? "cleanly encoded" : res.reliable ? "weak" : "unreliable (need more pairs)"}
            </span>
          </div>
          <PlainCard res={{ plain: res.plain }} />
        </>
      )}
    </div>
  );
}

export default function AnalysisPanel({ base }: { readonly base: string }): JSX.Element {
  return (
    <>
      <Causal base={base} />
      <Sae base={base} />
      <Modality />
      <Safety base={base} />
    </>
  );
}
