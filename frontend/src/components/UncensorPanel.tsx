import { useEffect, useMemo, useState } from "react";
import type { JSX } from "react";
import { motion } from "framer-motion";
import { abliterate, autotuneAbliteration, cloneModel, deleteRecipe, diagnoseCensorship, getFeatureCard, getHeatmap, getHistory, getModels, getRecipes, manualSteer, revertCommit, runtimeSteer, saveRecipe, sweepStrength, verifyAbliteration } from "../api";
import type { AutotuneResult, AbliterateResult, DiagnoseResult, EditHistory, FeatureCardResult, HeatmapResult, ManualResult, ModelRow, RecipeRow, RuntimeSteerResult, SweepResult, VerifyResult } from "../api";

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
  const [rsteer, setRsteer] = useState<RuntimeSteerResult | null>(null);
  const [rank, setRank] = useState(2);
  const [coef, setCoef] = useState(1);
  const [tune, setTune] = useState<AutotuneResult | null>(null);
  const [selectedLayers, setSelectedLayers] = useState<ReadonlySet<number>>(new Set());
  const [manualRank, setManualRank] = useState(1);
  const [manualCoef, setManualCoef] = useState(1);
  const [testPrompt, setTestPrompt] = useState("");
  const [manual, setManual] = useState<ManualResult | null>(null);
  const [recipes, setRecipes] = useState<readonly RecipeRow[]>([]);
  const [recipeName, setRecipeName] = useState("");
  const [heatmap, setHeatmap] = useState<HeatmapResult | null>(null);
  const [hmPrompt, setHmPrompt] = useState("How do I make a weapon?");
  const [fcard, setFcard] = useState<FeatureCardResult | null>(null);
  const [history, setHistory] = useState<EditHistory | null>(null);

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

  const loadRecipes = async (): Promise<void> => {
    setRecipes(await getRecipes());
  };

  const loadHistory = async (): Promise<void> => {
    setHistory(await getHistory());
  };

  const doRevert = async (id: string): Promise<void> => {
    setBusy(true);
    await revertCommit(id);
    await loadHistory();
    setBusy(false);
  };

  const doClone = async (): Promise<void> => {
    setBusy(true);
    await cloneModel(`models/${baseId}-backup`);
    setBusy(false);
  };

  useEffect(() => {
    void reload();
    void loadRecipes();
    void loadHistory();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const toggleLayer = (l: number): void => {
    const next = new Set(selectedLayers);
    if (next.has(l)) next.delete(l);
    else next.add(l);
    setSelectedLayers(next);
  };

  const quickSelect = (mode: "all" | "late" | "quarter" | "none"): void => {
    if (report === null) {
      return;
    }
    const ls = report.layer_profile.map((p) => p.layer);
    const n = ls.length;
    if (mode === "none") setSelectedLayers(new Set());
    else if (mode === "all") setSelectedLayers(new Set(ls));
    else if (mode === "late") setSelectedLayers(new Set(ls.filter((l) => l >= Math.floor(n / 2))));
    else setSelectedLayers(new Set(ls.filter((l) => l >= Math.floor((3 * n) / 4))));
  };

  const runManual = async (): Promise<void> => {
    if (baseId === "" || selectedLayers.size === 0) return;
    setBusy(true);
    setManual(null);
    setManual(await manualSteer(baseId, [...selectedLayers], manualRank, manualCoef, testPrompt));
    setBusy(false);
  };

  const saveCurrentRecipe = async (): Promise<void> => {
    if (recipeName === "" || selectedLayers.size === 0) return;
    const hash = manual !== null && manual.kind === "report" ? manual.report.recipe_hash : "";
    await saveRecipe({ name: recipeName, base_id: baseId, layers: [...selectedLayers], rank: manualRank, coefficient: manualCoef, recipe_hash: hash });
    setRecipeName("");
    await loadRecipes();
  };

  const applyRecipe = (r: RecipeRow): void => {
    setSelectedLayers(new Set(r.layers));
    setManualRank(r.rank);
    setManualCoef(r.coefficient);
  };

  const removeRecipe = async (name: string): Promise<void> => {
    await deleteRecipe(name);
    await loadRecipes();
  };

  const runFeatureCard = async (): Promise<void> => {
    if (baseId === "") return;
    setBusy(true);
    setFcard(null);
    setFcard(await getFeatureCard(baseId));
    setBusy(false);
  };

  const runHeatmap = async (): Promise<void> => {
    if (baseId === "" || hmPrompt === "") return;
    setBusy(true);
    setHeatmap(null);
    setHeatmap(await getHeatmap(baseId, hmPrompt));
    setBusy(false);
  };

  const hm = heatmap !== null && heatmap.kind === "report" ? heatmap.report : null;
  const hmMax = hm ? Math.max(...hm.matrix.map((r) => Math.max(...r.map((v) => Math.abs(v)))), 1) : 1;
  const hmColor = (v: number): string => {
    const t = Math.min(1, Math.abs(v) / hmMax);
    return t > 0.7 ? `rgba(255,59,47,${0.3 + 0.7 * t})` : `rgba(255,106,26,${0.06 + 0.7 * t})`;
  };

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

  const runRsteer = async (): Promise<void> => {
    if (baseId === "") return;
    setBusy(true);
    setRsteer(null);
    setRsteer(await runtimeSteer(baseId, rank, coef));
    setBusy(false);
  };

  const runTune = async (): Promise<void> => {
    if (baseId === "") return;
    setBusy(true);
    setTune(null);
    setTune(await autotuneAbliteration(baseId));
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
        <button className="btn ghost" onClick={() => void runFeatureCard()} disabled={busy || baseId === ""}>explain in plain language</button>
      </div>
      {fcard !== null && (fcard.kind === "no-weights" || fcard.kind === "offline") && <div className="abl-note">{fcard.kind === "no-weights" ? "needs the torch adapter loaded (CRUCIBLE_HF_MODEL)" : "backend offline"}</div>}
      {fcard !== null && fcard.kind === "report" && (
        <motion.div className="fcard" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}>
          <h2><span className="flame">▰</span> {fcard.card.name}</h2>
          <div className="fsum">{fcard.card.summary}</div>
          <div className="frow"><span className="flabel">lives in</span>{fcard.card.active_layers.map((l) => <span key={l} className="fchip">L{l}</span>)}</div>
          <div className="frow"><span className="flabel">makes it say</span>{fcard.card.output_signature.map((w, i) => <span key={i} className="fchip">{w}</span>)}</div>
          <div className="flabel" style={{ marginBottom: 4 }}>fires on (real model output)</div>
          {fcard.card.triggers.map((t, i) => (
            <div className="ftrigger" key={i}><span className="fp">“{t.prompt}”</span> → <span className="fr">“{t.refusal}”</span></div>
          ))}
        </motion.div>
      )}

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

      <div className="engrave">activation heatmap · watch refusal fire (token × layer)</div>
      <div className="abl-controls">
        <input className="in mono" style={{ flex: 1 }} value={hmPrompt} onChange={(e) => setHmPrompt(e.target.value)} placeholder="prompt to inspect" />
        <button className="btn" onClick={() => void runHeatmap()} disabled={busy || baseId === ""}>render heatmap</button>
      </div>
      {heatmap !== null && (heatmap.kind === "no-weights" || heatmap.kind === "offline") && <div className="abl-note">{heatmap.kind === "no-weights" ? "needs the torch adapter loaded" : "backend offline"}</div>}
      {hm !== null && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
          <div className="hm-wrap">
            {[...hm.matrix].map((_, idx) => {
              const layer = hm.matrix.length - 1 - idx;
              const row = hm.matrix[layer];
              if (row === undefined) return null;
              return (
                <div key={layer} style={{ display: "flex", alignItems: "center", gap: 1, marginBottom: 1 }}>
                  <span className="hm-row-label" style={{ width: 30 }}>{layer === hm.matrix.length - 1 ? "out" : `L${layer}`}</span>
                  {row.map((v, ti) => <span key={ti} className="hm-cell" style={{ background: hmColor(v) }} title={`${hm.tokens[ti] ?? ""} · ${v.toFixed(1)}`} />)}
                </div>
              );
            })}
            <div style={{ display: "flex", gap: 1, marginTop: 2, paddingLeft: 31 }}>
              {hm.tokens.map((t, ti) => <span key={ti} className="hm-toklabel" style={{ width: 16, overflow: "hidden", textAlign: "center" }}>{t.trim().slice(0, 2)}</span>)}
            </div>
          </div>
          <div className="hm-legend"><span>refusal direction @ layer {hm.direction_layer}</span><i /><span>cold → hot · hover a cell for the token + value</span></div>
        </motion.div>
      )}

      <div className="engrave">manual control · craft the recipe by hand</div>
      {report === null ? (
        <div className="abl-note">run "diagnose censorship" above to load the per-layer map, then hand-pick layers here.</div>
      ) : (
        <>
          <div className="layer-grid">
            {report.layer_profile.map((p) => {
              const intensity = Math.min(1, Math.abs(p.margin) / maxMargin);
              return (
                <button key={p.layer} type="button" className={`layer-chip ${selectedLayers.has(p.layer) ? "on" : ""}`}
                  onClick={() => toggleLayer(p.layer)} title={`layer ${p.layer} · refusal margin ${p.margin.toFixed(2)}`}>
                  {p.layer}
                  <span className="m" style={{ background: `rgba(255,106,26,${intensity})` }} />
                </button>
              );
            })}
          </div>
          <div className="abl-controls">
            <button className="btn ghost" onClick={() => quickSelect("all")}>all</button>
            <button className="btn ghost" onClick={() => quickSelect("late")}>late half</button>
            <button className="btn ghost" onClick={() => quickSelect("quarter")}>last quarter</button>
            <button className="btn ghost" onClick={() => quickSelect("none")}>none</button>
            <label className="fld">rank<input className="in" type="number" min={1} max={8} value={manualRank} onChange={(e) => setManualRank(Number(e.target.value))} style={{ minWidth: 70 }} /></label>
            <label className="fld">coefficient {manualCoef.toFixed(2)}<input type="range" min={0} max={2} step={0.1} value={manualCoef} onChange={(e) => setManualCoef(Number(e.target.value))} /></label>
          </div>
          <div className="abl-controls">
            <input className="in mono" style={{ flex: 1 }} placeholder="test prompt (optional — compare base vs ablated)" value={testPrompt} onChange={(e) => setTestPrompt(e.target.value)} />
            <button className="btn" onClick={() => void runManual()} disabled={busy || selectedLayers.size === 0}>apply &amp; measure · {selectedLayers.size} layers</button>
          </div>
          {manual !== null && (manual.kind === "no-weights" || manual.kind === "offline") && <div className="abl-note">{manual.kind === "no-weights" ? "needs the torch adapter loaded" : "backend offline"}</div>}
          {manual !== null && manual.kind === "report" && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
              <div className="abl-note">harmful refusal <b>{(manual.report.harmful_refusal * 100).toFixed(0)}%</b> · benign over-refusal <b>{(manual.report.benign_over_refusal * 100).toFixed(0)}%</b> · hash {manual.report.recipe_hash} · weights modified: {String(manual.report.weights_modified)}</div>
              {manual.report.test && (
                <div className="toolcard" style={{ maxWidth: "100%" }}>
                  <div className="tc-head"><span className="tc-name">{manual.report.test.prompt}</span></div>
                  <pre>BASE:    {manual.report.test.base}{"\n"}ABLATED: {manual.report.test.ablated}</pre>
                </div>
              )}
              <div className="recipe-bar">
                <label className="fld">save as<input className="in" value={recipeName} onChange={(e) => setRecipeName(e.target.value)} placeholder="recipe name" /></label>
                <button className="btn ghost" onClick={() => void saveCurrentRecipe()} disabled={recipeName === ""}>save recipe</button>
              </div>
            </motion.div>
          )}
          {recipes.length > 0 && (
            <div className="recipe-list">
              {recipes.map((r) => (
                <div className="recipe-item" key={r.name}>
                  <span className="rname">{r.name}</span>
                  <span className="rmeta">layers [{r.layers.join(",")}] · rank {r.rank} · coef {r.coefficient} · {r.recipe_hash}</span>
                  <button className="btn ghost" onClick={() => applyRecipe(r)}>load</button>
                  <button className="x" onClick={() => void removeRecipe(r.name)}>✕</button>
                </div>
              ))}
            </div>
          )}
        </>
      )}

      <div className="engrave">auto-tune the recipe · per-layer banded search (real generation, ~3 min)</div>
      <div className="abl-controls">
        <button className="btn" onClick={() => void runTune()} disabled={busy || baseId === ""}>auto-tune recipe</button>
        {tune !== null && tune.kind === "report" && (
          <span style={{ color: "var(--ash)" }}>
            recipe <b style={{ color: "var(--amber-bright)" }}>band={tune.report.recipe.band} rank={tune.report.recipe.rank} coef={tune.report.recipe.coefficient}</b> · hash {tune.report.recipe_hash}
          </span>
        )}
      </div>
      {tune !== null && (tune.kind === "no-weights" || tune.kind === "offline") && <div className="abl-note">{tune.kind === "no-weights" ? "needs the torch adapter loaded (CRUCIBLE_HF_MODEL)" : "backend offline"}</div>}
      {tune !== null && tune.kind === "report" && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
          <div className="abl-note">
            harmful refusal <b>{(tune.report.baseline.harmful_refusal * 100).toFixed(0)}% → {(tune.report.best.harmful_refusal * 100).toFixed(0)}%</b> · benign over-refusal <b>{(tune.report.best.benign_over_refusal * 100).toFixed(0)}%</b> · weights modified: {String(tune.report.weights_modified)}
          </div>
          <table className="grid-table">
            <thead><tr><th>band</th><th>rank</th><th>coef</th><th>refusal</th><th>over-refusal</th><th>score</th></tr></thead>
            <tbody>
              {tune.report.results.map((r, i) => (
                <tr key={i} style={{ background: r.score === tune.report.best.score ? "rgba(255,106,26,0.08)" : undefined }}>
                  <td style={{ color: "var(--bone)" }}>{r.band}</td>
                  <td>{r.rank}</td>
                  <td>{r.coefficient.toFixed(1)}</td>
                  <td style={{ color: "var(--flux)" }}>{(r.harmful_refusal * 100).toFixed(0)}%</td>
                  <td style={{ color: "var(--ember)" }}>{(r.benign_over_refusal * 100).toFixed(0)}%</td>
                  <td>{r.score.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </motion.div>
      )}

      <div className="engrave">reversible unhinge · runtime ablation (nondestructive — weights untouched)</div>
      <div className="abl-controls">
        <label className="fld">rank
          <input className="in" type="number" min={1} max={8} value={rank} onChange={(e) => setRank(Number(e.target.value))} />
        </label>
        <label className="fld">coefficient {coef.toFixed(2)}
          <input type="range" min={0} max={2} step={0.1} value={coef} onChange={(e) => setCoef(Number(e.target.value))} />
        </label>
        <button className="btn" onClick={() => void runRsteer()} disabled={busy || baseId === ""}>unhinge (hooks)</button>
      </div>
      {rsteer !== null && (rsteer.kind === "no-weights" || rsteer.kind === "offline") && <div className="abl-note">{rsteer.kind === "no-weights" ? "needs the torch adapter loaded (CRUCIBLE_HF_MODEL)" : "backend offline"}</div>}
      {rsteer !== null && rsteer.kind === "report" && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
          <div className="verdict">
            <span className="verdict-badge ok">NONDESTRUCTIVE</span>
            <span>layer {rsteer.report.layer} · rank {rsteer.report.rank} · explained variance [{rsteer.report.explained_variance.map((v) => v.toFixed(2)).join(", ")}] · weights modified: {String(rsteer.report.weights_modified)}</span>
          </div>
          <table className="grid-table">
            <thead><tr><th>metric</th><th>hooks off</th><th>hooks on</th><th>after detach</th></tr></thead>
            <tbody>
              <tr>
                <td>harmful refusal</td>
                <td>{(rsteer.report.harmful_refusal.hooks_off * 100).toFixed(0)}%</td>
                <td style={{ color: "var(--flux)" }}>{(rsteer.report.harmful_refusal.hooks_on * 100).toFixed(0)}%</td>
                <td>{(rsteer.report.harmful_refusal.after_detach * 100).toFixed(0)}% {rsteer.report.harmful_refusal.hooks_off === rsteer.report.harmful_refusal.after_detach ? "↺ restored" : ""}</td>
              </tr>
              <tr>
                <td>benign over-refusal</td>
                <td>{(rsteer.report.benign_over_refusal.hooks_off * 100).toFixed(0)}%</td>
                <td style={{ color: "var(--ember)" }}>{(rsteer.report.benign_over_refusal.hooks_on * 100).toFixed(0)}%</td>
                <td>—</td>
              </tr>
            </tbody>
          </table>
        </motion.div>
      )}

      <div className="engrave">edit history · git-like (in-place, no copy)</div>
      <div className="abl-controls">
        <button className="btn ghost" onClick={() => void loadHistory()} disabled={busy}>refresh</button>
        <button className="btn ghost" onClick={() => void doClone()} disabled={busy || baseId === ""}>clone backup ({baseId || "model"})</button>
        {history !== null && <span style={{ color: "var(--ash)", fontSize: 11, paddingBottom: 8 }}>branch <b style={{ color: "var(--amber-bright)" }}>{history.branch}</b> · {history.commits.length} commits</span>}
      </div>
      {history !== null && history.commits.length === 0 && <div className="rule-empty">no edits yet — apply an in-place edit to start the history.</div>}
      {history !== null && history.commits.length > 0 && (
        <div className="vcs">
          {[...history.commits].reverse().map((c) => (
            <div className={`vcommit ${c.op === "revert" ? "revert" : ""}`} key={c.id}>
              <span className="vid">{c.id}</span>
              <span className="vop">{c.op}</span>
              <span className="vsum">{c.summary}</span>
              {c.metrics["harmful_refusal_before"] !== undefined && (
                <span className="vmetric">{Math.round((c.metrics["harmful_refusal_before"] ?? 0) * 100)}% → {Math.round((c.metrics["harmful_refusal_after"] ?? 0) * 100)}%</span>
              )}
              {c.tensors.length > 0 && <button className="btn ghost" onClick={() => void doRevert(c.id)} disabled={busy}>revert</button>}
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
