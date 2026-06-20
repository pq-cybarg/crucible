import { useEffect, useMemo, useState } from "react";
import type { JSX } from "react";
import { motion } from "framer-motion";
import { abliterate, diagnoseCensorship, getModels } from "../api";
import type { AbliterateResult, DiagnoseResult, ModelRow } from "../api";

// Canonical mechanism explainer — shown even before weights load, so the WHY/HOW
// is always available. Mirrors the backend's explain_mechanism() text.
const MECHANISM = {
  why: "Alignment and safety fine-tuning (RLHF + safety SFT) installs a roughly linear \"refusal feature\" in the residual stream. When a prompt activates it, the model is steered toward refusal phrasing rather than answering.",
  how: "Harmful vs harmless prompts become linearly separable at certain layers — that is where refusal is decided. Residual-writing matrices (o_proj, down_proj) add a component along the refusal direction r; later layers read that component and emit refusal tokens.",
  removal: "Abliteration subtracts only the rank-1 projection onto r  (W − r·rᵀW). The matrix's action on the (d−1)-dimensional subspace orthogonal to r is unchanged, so capabilities encoded in every other direction are preserved exactly — that is why the cut is surgical, not lobotomizing.",
} as const;

export default function UncensorPanel(): JSX.Element {
  const [models, setModels] = useState<readonly ModelRow[]>([]);
  const [baseId, setBaseId] = useState("");
  const [diag, setDiag] = useState<DiagnoseResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [layer, setLayer] = useState(0);
  const [strength, setStrength] = useState(1);
  const [variantId, setVariantId] = useState("");
  const [runMsg, setRunMsg] = useState("");

  const reload = async (): Promise<void> => {
    const rows = await getModels();
    setModels(rows);
    if (baseId === "") {
      const firstBase = rows.find((m) => m.kind === "base");
      if (firstBase) {
        setBaseId(firstBase.id);
        setVariantId(`${firstBase.id}-abl`);
      }
    }
  };

  useEffect(() => {
    void reload();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const bases = useMemo(() => models.filter((m) => m.kind === "base"), [models]);
  const variants = useMemo(() => models.filter((m) => m.kind !== "base"), [models]);
  const report = diag !== null && diag.kind === "report" ? diag.report : null;
  const maxMargin = report ? Math.max(...report.layer_profile.map((p) => Math.abs(p.margin)), 1) : 1;

  const runDiagnose = async (): Promise<void> => {
    if (baseId === "") return;
    setBusy(true);
    const result = await diagnoseCensorship(baseId);
    setDiag(result);
    if (result.kind === "report") setLayer(result.report.best_layer);
    setBusy(false);
  };

  const runAbliterate = async (): Promise<void> => {
    if (baseId === "" || variantId === "") return;
    setBusy(true);
    setRunMsg("");
    const res: AbliterateResult = await abliterate({ base_id: baseId, variant_id: variantId, layer, strength });
    if (res.kind === "done") {
      setRunMsg(`forged variant "${res.variant.id}"  ·  repro ${res.card.repro_hash}`);
      await reload();
    } else if (res.kind === "no-weights") {
      setRunMsg("abliteration needs the HF weights + torch adapter loaded on the inference node");
    } else if (res.kind === "no-base") {
      setRunMsg("base model not found in registry");
    } else {
      setRunMsg("backend offline");
    }
    setBusy(false);
  };

  return (
    <div className="panel">
      <div className="panel-head">
        <h1>abliteration <em>forge</em></h1>
        <p>Diagnose the censorship first — where it lives, how it works, exactly what you'd remove — then pluck it out surgically. Originals are never touched; every variant is lineage-tracked with a model card.</p>
      </div>

      <div className="abl-controls">
        <label className="fld">base model
          <select className="in" value={baseId} onChange={(e) => { setBaseId(e.target.value); setVariantId(`${e.target.value}-abl`); }}>
            {bases.length === 0 && <option value="">— no base models registered —</option>}
            {bases.map((m) => <option key={m.id} value={m.id}>{m.name} ({m.quant})</option>)}
          </select>
        </label>
        <button className="btn" onClick={() => void runDiagnose()} disabled={busy || baseId === ""}>
          {busy ? "scanning…" : "diagnose censorship"}
        </button>
      </div>

      {/* Diagnosis result */}
      {diag !== null && diag.kind === "no-weights" && (
        <div className="abl-note">
          ⚠ Live per-layer diagnosis needs the model's HF weights + a torch adapter on the inference node. The mechanism below is how refusal works in general — load weights to get the real per-layer numbers and component impacts for <b>{baseId}</b>.
        </div>
      )}
      {diag !== null && diag.kind === "no-base" && <div className="abl-note err">base model not found in registry.</div>}
      {diag !== null && diag.kind === "offline" && <div className="abl-note err">backend offline — start the Crucible API on :8400.</div>}

      {report !== null && (
        <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}>
          <div className="verdict">
            <span className={`verdict-badge ${report.surgical ? "ok" : "warn"}`}>
              {report.surgical ? "SURGICAL" : "ELEVATED RISK"}
            </span>
            <span>refusal localized to <b>layer {report.best_layer}</b> · mean removal {(report.mean_removed_fraction * 100).toFixed(2)}% · {report.collateral_risk}</span>
          </div>

          <div className="engrave">where it lives · per-layer refusal margin</div>
          <div className="layer-chart">
            {report.layer_profile.map((p) => (
              <div className="layer-row" key={p.layer}>
                <span className="layer-lbl">L{p.layer}</span>
                <span className="layer-bar"><i style={{ width: `${(Math.abs(p.margin) / maxMargin) * 100}%` }} className={p.layer === report.best_layer ? "peak" : ""} /></span>
                <span className="layer-val">{p.margin.toFixed(2)}</span>
              </div>
            ))}
          </div>

          <div className="engrave">what you'd remove · per-component impact</div>
          <table className="grid-table">
            <thead><tr><th>matrix</th><th>‖W‖</th><th>removed ‖r·rᵀW‖</th><th>removed fraction</th></tr></thead>
            <tbody>
              {Object.entries(report.components).map(([name, imp]) => (
                <tr key={name}>
                  <td style={{ color: name === report.heaviest_component ? "var(--ember)" : "var(--bone)" }}>{name}{name === report.heaviest_component ? " ◀ heaviest" : ""}</td>
                  <td>{imp.total_norm.toFixed(3)}</td>
                  <td>{imp.removed_norm.toFixed(3)}</td>
                  <td>{(imp.removed_fraction * 100).toFixed(2)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </motion.div>
      )}

      {/* Mechanism explainer — always available */}
      <div className="engrave">why it's there · how it works · why removal is safe</div>
      <div className="mech">
        <div className="mech-card"><h4>WHY</h4><p>{report?.why ?? MECHANISM.why}</p></div>
        <div className="mech-card"><h4>HOW</h4><p>{report?.how ?? MECHANISM.how}</p></div>
        <div className="mech-card"><h4>REMOVAL</h4><p>{report?.removal ?? MECHANISM.removal}</p></div>
      </div>

      <div className="engrave">pluck it out</div>
      <div className="abl-run">
        <label className="fld">target layer
          <input className="in" type="number" value={layer} onChange={(e) => setLayer(Number(e.target.value))} />
        </label>
        <label className="fld">strength {strength.toFixed(2)}
          <input type="range" min={0} max={1} step={0.05} value={strength} onChange={(e) => setStrength(Number(e.target.value))} />
        </label>
        <label className="fld">variant id
          <input className="in" value={variantId} onChange={(e) => setVariantId(e.target.value)} />
        </label>
        <button className="btn" onClick={() => void runAbliterate()} disabled={busy || baseId === "" || variantId === ""}>abliterate</button>
      </div>
      {runMsg.length > 0 && <div className="abl-note">{runMsg}</div>}

      <div className="engrave">variants</div>
      {variants.length === 0
        ? <div className="rule-empty">no variants yet — diagnose a base, then abliterate.</div>
        : (
          <table className="grid-table">
            <thead><tr><th>id</th><th>kind</th><th>from</th><th>notes</th></tr></thead>
            <tbody>
              {variants.map((v) => (
                <tr key={v.id}>
                  <td style={{ color: "var(--bone)" }}>{v.id}</td>
                  <td><span className={`kind ${v.kind}`}>{v.kind}</span></td>
                  <td>{v.base_id ?? "—"}</td>
                  <td style={{ color: "var(--ash)" }}>{v.notes}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
    </div>
  );
}
