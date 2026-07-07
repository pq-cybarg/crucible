import type { JSX } from "react";
import { useEffect, useState } from "react";
import {
  BUILTIN_TOOLS, PATH_TOOLS, getMetrics, getModels, getPreferences, getToolNames, savePreferences,
} from "../api";
import type {
  MetricInfo, ModelRow, PathRuleConfig, PermissionMode, Preferences,
} from "../api";

// The workbench's preference center: HOW recall is ordered & measured, WHICH model does the
// background reorganization, and WHAT the agent's tools are allowed to touch. Everything here is a
// persisted default the rest of the app reads — the forge seeds its permissions from the tool-
// permission block, and memory search/index honor the sort + metric defaults.
const MODES: readonly PermissionMode[] = ["allow", "ask", "deny"];
const SORT_HELP: Record<string, string> = {
  recency: "newest first (recency bias)", oldest: "oldest first (primacy bias)",
  priority: "most salient first (what you weighted)", balanced: "blend recency + salience (human-recall)",
  relevance: "best query match first", size: "biggest first", degree: "most connected first", label: "A→Z",
};

export default function PreferencesPanel(): JSX.Element {
  const [prefs, setPrefs] = useState<Preferences | null>(null);
  const [sorts, setSorts] = useState<readonly string[]>([]);
  const [metrics, setMetrics] = useState<readonly MetricInfo[]>([]);
  const [models, setModels] = useState<readonly ModelRow[]>([]);
  const [tools, setTools] = useState<readonly string[]>([...BUILTIN_TOOLS]);
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => { void (async () => {
    try {
      const [p, mc, ms, tn] = await Promise.all([
        getPreferences(), getMetrics(), getModels().catch(() => []), getToolNames(),
      ]);
      setPrefs(p.preferences); setSorts(p.sorts); setMetrics(mc.metrics); setModels(ms); setTools(tn);
    } catch (e: unknown) { setErr(e instanceof Error ? e.message : "failed to load preferences"); }
  })(); }, []);

  function patch(p: Partial<Preferences>): void { setPrefs((cur) => (cur ? { ...cur, ...p } : cur)); }
  function patchPerms(p: Partial<Preferences["permissions"]>): void {
    setPrefs((cur) => (cur ? { ...cur, permissions: { ...cur.permissions, ...p } } : cur));
  }
  function setToolMode(tool: string, mode: PermissionMode | ""): void {
    if (!prefs) return;
    const modes = { ...prefs.permissions.modes };
    if (mode === "") delete modes[tool]; else modes[tool] = mode;
    patchPerms({ modes });
  }
  const rules = (): readonly PathRuleConfig[] => prefs?.permissions.path_rules ?? [];
  function setRule(i: number, r: Partial<PathRuleConfig>): void {
    patchPerms({ path_rules: rules().map((x, j) => (j === i ? { ...x, ...r } : x)) });
  }
  function addRule(): void { patchPerms({ path_rules: [...rules(), { glob: "", mode: "deny", tools: [] }] }); }
  function delRule(i: number): void { patchPerms({ path_rules: rules().filter((_, j) => j !== i) }); }

  async function save(): Promise<void> {
    if (!prefs) return;
    setBusy(true); setErr(null); setNote(null);
    try {
      // drop empty-glob rules so a half-typed row isn't persisted
      const clean = { ...prefs, permissions: { ...prefs.permissions, path_rules: rules().filter((r) => r.glob.trim().length > 0) } };
      const saved = await savePreferences(clean);
      setPrefs(saved); setNote("preferences saved");
    } catch (e: unknown) { setErr(e instanceof Error ? e.message : "save failed"); } finally { setBusy(false); }
  }

  if (!prefs) {
    return <div className="panel"><div className="panel-head"><h1><em>preferences</em></h1></div>
      <div className="hint">{err ?? "loading…"}</div></div>;
  }
  const perms = prefs.permissions;
  const llmMetricChosen = prefs.default_metric === "llm";

  return (
    <div className="panel">
      <div className="panel-head">
        <h1>work<em>bench</em> preferences</h1>
        <p>Defaults for how memory is <b>recalled</b> and how <b>closeness</b> is measured, which model
          does background <b>reorganization</b>, and what your agent's tools may <b>touch</b>. These
          persist and drive the rest of the app — the forge seeds its permissions from here.</p>
      </div>

      {/* --- recall & distance ------------------------------------------------------------------ */}
      <div className="pref-section">
        <div className="engrave">recall &amp; distance</div>
        <div className="pref-grid">
          <label className="fld">default recall order
            <select className="in" value={prefs.default_sort} onChange={(e) => patch({ default_sort: e.target.value })}>
              {sorts.map((s) => <option key={s} value={s}>{s}{SORT_HELP[s] ? ` — ${SORT_HELP[s]}` : ""}</option>)}
            </select>
          </label>
          <label className="fld">default distance metric
            <select className="in" value={prefs.default_metric} onChange={(e) => patch({ default_metric: e.target.value })}>
              {metrics.map((m) => (
                <option key={m.name} value={m.name} disabled={!m.available}>
                  {m.name} — {m.label}{m.available ? "" : " (needs backend)"}
                </option>
              ))}
            </select>
          </label>
        </div>

        {prefs.default_sort === "balanced" && (
          <label className="fld pref-weight">balanced blend — recency&nbsp;{Math.round(prefs.balanced_recency_weight * 100)}%
            &nbsp;/&nbsp;salience&nbsp;{Math.round((1 - prefs.balanced_recency_weight) * 100)}%
            <input type="range" min={0} max={1} step={0.05} value={prefs.balanced_recency_weight}
              onChange={(e) => patch({ balanced_recency_weight: Number(e.target.value) })} />
          </label>
        )}

        <label className="fld">processing model <span className="hint" style={{ margin: 0 }}>— the small/cheap model for llm-judged distance &amp; background reorganization (the plasticity role)</span>
          <select className="in" value={prefs.processing_model ?? ""} onChange={(e) => patch({ processing_model: e.target.value || null })}>
            <option value="">none — llm-judged distance disabled</option>
            {models.map((m) => <option key={m.id} value={m.id}>{m.name}</option>)}
          </select>
        </label>
        {llmMetricChosen && !prefs.processing_model && (
          <div className="runtime-err">llm-judged is the default metric but no processing model is set — search will fall back to lexical.</div>
        )}
        <p className="hint" style={{ marginTop: 6 }}>
          Metrics are labeled by the KIND of closeness they measure — <b>statistical</b> (token/character
          overlap), <b>lexical</b> (BM25 keywords), <b>semantic</b> (real embeddings), <b>llm-judged</b>
          (a model's opinion). A keyword or bag-of-words hit is never presented as meaning.
        </p>
      </div>

      {/* --- tool permissions ------------------------------------------------------------------- */}
      <div className="pref-section">
        <div className="engrave">tool permissions</div>
        <label className="fld">default for every tool
          <span className="seg">
            {MODES.map((m) => (
              <button type="button" key={m} className={perms.default === m ? "on" : ""}
                onClick={() => patchPerms({ default: m })}>{m}</button>
            ))}
          </span>
        </label>

        <div className="engrave" style={{ marginTop: 10 }}>per-tool overrides</div>
        <div className="pref-tools">
          {tools.map((t) => {
            const cur = perms.modes[t];
            return (
              <div className="pref-tool" key={t}>
                <code className="mem-key">{t}</code>
                <span className="seg">
                  <button type="button" className={cur === undefined ? "on" : ""} onClick={() => setToolMode(t, "")}>default</button>
                  {MODES.map((m) => (
                    <button type="button" key={m} className={cur === m ? "on" : ""} onClick={() => setToolMode(t, m)}>{m}</button>
                  ))}
                </span>
              </div>
            );
          })}
        </div>

        <div className="engrave" style={{ marginTop: 10 }}>path rules
          <span className="hint" style={{ margin: 0 }}> — limit tools to specific files/directories (firewall order, deny wins)</span>
        </div>
        <div className="pref-rules">
          {rules().length === 0 && <div className="hint">no path rules — add one to e.g. deny <code>~/.ssh/**</code> or allow a safe workspace.</div>}
          {rules().map((r, i) => (
            <div className="pref-rule" key={i}>
              <input className="in" placeholder="glob, e.g. ~/.ssh/**  or  /work/**" value={r.glob}
                onChange={(e) => setRule(i, { glob: e.target.value })} />
              <span className="seg">
                {MODES.map((m) => (
                  <button type="button" key={m} className={r.mode === m ? "on" : ""} onClick={() => setRule(i, { mode: m })}>{m}</button>
                ))}
              </span>
              <select className="in pref-rule-tools" multiple value={[...r.tools]}
                title="tools this rule applies to (none = all path-taking tools)"
                onChange={(e) => setRule(i, { tools: Array.from(e.target.selectedOptions, (o) => o.value) })}>
                {PATH_TOOLS.map((t) => <option key={t} value={t}>{t}</option>)}
              </select>
              <button className="hier-del" title="remove rule" onClick={() => delRule(i)}>✕</button>
            </div>
          ))}
          <button className="btn ghost" onClick={addRule}>+ path rule</button>
        </div>
      </div>

      <div className="pref-actions">
        <button className="btn" disabled={busy} onClick={() => void save()}>save preferences</button>
        {note && <span className="mem-note">{note}</span>}
        {err && <span className="runtime-err">{err}</span>}
      </div>
    </div>
  );
}
