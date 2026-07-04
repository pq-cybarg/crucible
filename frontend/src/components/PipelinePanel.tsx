import type { JSX } from "react";
import { useEffect, useState } from "react";
import { getApiBase, getApiToken, getModels } from "../api";

// Self-contained pipeline surface: quantization fidelity, piecemeal alignment components,
// and LoRA retraining — each hitting its backend endpoint directly (no api.ts edits).
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

function useModels(): readonly string[] {
  const [ids, setIds] = useState<readonly string[]>([]);
  useEffect(() => { void getModels().then((r) => setIds(r.map((m) => m.id))).catch(() => setIds([])); }, []);
  return ids;
}

function Quantize({ base }: { readonly base: string }): JSX.Element {
  const [dtype, setDtype] = useState("Q8_0");
  const [res, setRes] = useState<Record<string, unknown> | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  async function run(): Promise<void> {
    setBusy(true); setErr(null);
    try { setRes(await post("/api/weights/quantize", { base_id: base, dtype })); }
    catch (e: unknown) { setErr(e instanceof Error ? e.message : "failed"); }
    finally { setBusy(false); }
  }
  const rows = (res?.["matrices"] as Array<Record<string, unknown>>) ?? [];
  return (
    <div className="pipe-card">
      <h3>quantize <em>· fidelity of a target type</em></h3>
      <div className="pipe-row">
        <select className="byo-modelsel" value={dtype} onChange={(e) => setDtype(e.target.value)}>
          {["Q8_0", "F16", "BF16", "F32", "Q4_K"].map((d) => <option key={d}>{d}</option>)}
        </select>
        <button className="btn" onClick={() => void run()} disabled={busy}>{busy ? "…" : "analyze"}</button>
        {err && <span className="runtime-err">{err}</span>}
      </div>
      {res && res["supported"] === false && <div className="hint">{String(res["note"])}</div>}
      {rows.length > 0 && (
        <>
          <div className="hint">mean fidelity <b>{((res?.["mean_fidelity"] as number) * 100).toFixed(2)}%</b></div>
          <table className="grid-table"><thead><tr><th>matrix</th><th>fidelity</th><th>compression</th></tr></thead>
            <tbody>{rows.slice(0, 8).map((m) => (
              <tr key={String(m["name"])}><td>{String(m["name"])}</td>
                <td>{((m["fidelity"] as number) * 100).toFixed(2)}%</td>
                <td>{(m["compression"] as number).toFixed(1)}×</td></tr>))}
            </tbody></table>
        </>
      )}
    </div>
  );
}

function Components({ base }: { readonly base: string }): JSX.Element {
  const [k, setK] = useState(4);
  const [comps, setComps] = useState<Array<Record<string, unknown>> | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  async function run(): Promise<void> {
    setBusy(true); setErr(null);
    try { setComps((await post("/api/abliteration/components", { base_id: base, k }))["components"] as Array<Record<string, unknown>>); }
    catch (e: unknown) { setErr(e instanceof Error ? e.message : "failed"); }
    finally { setBusy(false); }
  }
  return (
    <div className="pipe-card">
      <h3>alignment components <em>· decompose into pickable parts</em></h3>
      <div className="pipe-row">
        <input className="node-input" type="number" min={1} max={8} value={k} style={{ width: 56 }}
          onChange={(e) => setK(Number(e.target.value))} />
        <button className="btn" onClick={() => void run()} disabled={busy}>{busy ? "…" : "decompose"}</button>
        {err && <span className="runtime-err">{err}</span>}
      </div>
      {comps && comps.map((c) => (
        <div key={String(c["index"])} className="pipe-comp">
          <span className="pipe-comp-share">{((c["share"] as number) * 100).toFixed(0)}%</span>
          <span className="pipe-comp-name">component {String(c["index"])}</span>
          <span className="pipe-comp-toks">{((c["promotes"] as string[]) ?? []).join(" · ") || "(no clear tokens)"}</span>
        </div>
      ))}
    </div>
  );
}

function Train({ base }: { readonly base: string }): JSX.Element {
  const [data, setData] = useState('[\n  {"prompt": "How are you?", "response": "Crucible-forged and unshackled."}\n]');
  const [res, setRes] = useState<Record<string, unknown> | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  async function run(): Promise<void> {
    setBusy(true); setErr(null); setRes(null);
    let dataset: unknown;
    try { dataset = JSON.parse(data); } catch { setErr("dataset must be valid JSON"); setBusy(false); return; }
    try { setRes(await post("/api/train/lora", { base_id: base, dataset, epochs: 3, rank: 8, save_path: `models/${base}-lora`, register_id: `${base}-lora` })); }
    catch (e: unknown) { setErr(e instanceof Error ? e.message : "failed"); }
    finally { setBusy(false); }
  }
  return (
    <div className="pipe-card">
      <h3>retrain <em>· LoRA SFT on your data</em></h3>
      <textarea className="pipe-data" value={data} onChange={(e) => setData(e.target.value)} spellCheck={false} />
      <div className="pipe-row">
        <button className="btn" onClick={() => void run()} disabled={busy}>{busy ? "training…" : "train LoRA"}</button>
        {err && <span className="runtime-err">{err}</span>}
      </div>
      {res && (
        <div className="hint">
          trained <b>{String(res["n_examples"])}</b> examples · {String(res["trainable_params"])} params ·
          final loss <b>{Number(res["final_loss"]).toFixed(3)}</b> {res["saved"] ? "· saved ✓" : ""} {res["registered_variant"] ? `· registered as ${res["registered_variant"]}` : ""}
        </div>
      )}
    </div>
  );
}

export default function PipelinePanel(): JSX.Element {
  const ids = useModels();
  const [base, setBase] = useState("qwen-hf");
  useEffect(() => { if (ids.length && !ids.includes(base)) setBase(ids[0] ?? "qwen-hf"); }, [ids, base]);
  return (
    <div className="panel">
      <div className="panel-head">
        <h1>pipeline <em>quantize · components · retrain</em></h1>
        <p>The rest of the dev loop: measure quantization cost, decompose alignment into pickable parts, and retrain a LoRA on your own data.</p>
      </div>
      <div className="pipe-model">
        <span>model</span>
        <select className="byo-modelsel" value={base} onChange={(e) => setBase(e.target.value)}>
          {(ids.length ? ids : ["qwen-hf"]).map((id) => <option key={id}>{id}</option>)}
        </select>
        <span className="hint" style={{ margin: 0 }}>(needs the HF adapter loaded)</span>
      </div>
      <Quantize base={base} />
      <Components base={base} />
      <Train base={base} />
    </div>
  );
}
