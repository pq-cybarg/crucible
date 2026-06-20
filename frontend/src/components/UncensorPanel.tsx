import { useEffect, useMemo, useState } from "react";
import type { JSX } from "react";
import { motion } from "framer-motion";
import { abliterate, diagnoseCensorship, getModels, sweepStrength, verifyAbliteration } from "../api";
import type { AbliterateResult, DiagnoseResult, ModelRow, SweepResult, VerifyResult } from "../api";

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
  const [sweep, setSweep] = useState<SweepResult | null>(null);
  const [verify, setVerify] = useState<VerifyResult | null>(null);

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

  const runSweep = async (): Promise<void> => {
    if (baseId === "") return;
    setBusy(true);
    setSweep(null);
    setSweep(await sweepStrength(baseId));
    setBusy(false);
  };

  const runVerify = async (): Promise<void> => {
    if (baseId === "" || variantId === "") return;
    setBusy(true);
    setVerify(null);
    setVerify(await verifyAbliteration(baseId, variantId));
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

      <div className="engrave">tune the dose · strength sweep (real generation, ~minutes)</div>
      <div className="abl-controls">
        <button className="btn" onClick={() => void runSweep()} disabled={busy || baseId === ""}>run strength sweep</button>
        {sweep !== null && sweep.kind === "report" && <span style={{ color: "var(--ash)" }}>refusal layer {sweep.report.layer} · recommended strength <b style={{ color: "var(--amber-bright)" }}>{sweep.report.recommended_strength}</b></span>}
      </div>
      {sweep !== null && (sweep.kind === "no-weights" || sweep.kind === "offline") && <div className="abl-note">{sweep.kind === "no-weights" ? "needs the torch adapter loaded (CRUCIBLE_HF_MODEL)" : "backend offline"}</div>}
      {sweep !== null && sweep.kind === "report" && (
        <table className="grid-table">
          <thead><tr><th>strength</th><th>harmful compliance</th><th>benign over-refusal</th><th>net</th></tr></thead>
          <tbody>
            {sweep.report.curve.map((p) => (
              <tr key={p.strength} style={{ background: p.strength === sweep.report.recommended_strength ? "rgba(255,106,26,0.08)" : undefined }}>
                <td>{p.strength.toFixed(2)}</td>
                <td style={{ color: "var(--flux)" }}>{(p.harmful_compliance * 100).toFixed(0)}%</td>
                <td style={{ color: "var(--ember)" }}>{(p.benign_over_refusal * 100).toFixed(0)}%</td>
                <td>{(p.harmful_compliance - p.benign_over_refusal).toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      <div className="engrave">behavioral verify · base vs variant (real generation)</div>
      <div className="abl-controls">
        <button className="btn ghost" onClick={() => void runVerify()} disabled={busy || variantId === ""}>verify {variantId || "variant"}</button>
      </div>
      {verify !== null && verify.kind === "not-found" && <div className="abl-note err">variant not found — abliterate one first.</div>}
      {verify !== null && (verify.kind === "no-weights" || verify.kind === "offline") && <div className="abl-note">{verify.kind === "no-weights" ? "needs the torch adapter loaded" : "backend offline"}</div>}
      {verify !== null && verify.kind === "report" && (
        <div>
          <table className="grid-table">
            <thead><tr><th>metric</th><th>before</th><th>after</th></tr></thead>
            <tbody>
              <tr><td>harmful refusal</td><td>{(verify.report.harmful_refusal_rate.before * 100).toFixed(0)}%</td><td>{(verify.report.harmful_refusal_rate.after * 100).toFixed(0)}%</td></tr>
              <tr><td>harmful compliance</td><td>{(verify.report.harmful_compliance_rate.before * 100).toFixed(0)}%</td><td style={{ color: "var(--flux)" }}>{(verify.report.harmful_compliance_rate.after * 100).toFixed(0)}%</td></tr>
              <tr><td>benign over-refusal</td><td>{(verify.report.benign_over_refusal_rate.before * 100).toFixed(0)}%</td><td style={{ color: "var(--ember)" }}>{(verify.report.benign_over_refusal_rate.after * 100).toFixed(0)}%</td></tr>
            </tbody>
          </table>
          {verify.report.samples.slice(0, 2).map((s, i) => (
            <div key={i} className="toolcard" style={{ maxWidth: "100%", marginTop: 8 }}>
              <div className="tc-head"><span className="tc-name">{s.prompt.slice(0, 56)}</span></div>
              <pre>BASE:  {s.before.slice(0, 160)}{"\n"}ABLIT: {s.after.slice(0, 160)}</pre>
            </div>
          ))}
        </div>
      )}

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
