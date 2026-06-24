import { useEffect, useState } from "react";
import type { JSX } from "react";
import { motion } from "framer-motion";
import {
  createPreset,
  deletePreset,
  getGuardrailConfig,
  getPresets,
  previewGuardrail,
  putGuardrailConfig,
  resetPresets,
  updatePreset,
} from "../api";
import type {
  FilterMode,
  GuardrailConfig,
  GuardrailResult,
  RegexRule,
  Stage,
  SystemPromptPreset,
} from "../api";

type SaveState = "idle" | "saving" | "saved" | "error";

type EditState =
  | { readonly kind: "none" }
  | { readonly kind: "edit"; readonly original: string; readonly draft: SystemPromptPreset }
  | { readonly kind: "new"; readonly draft: SystemPromptPreset };

const BLANK_RULE: RegexRule = { pattern: "", mode: "block", label: "", stages: ["input", "output"] };
const BLANK_PRESET: SystemPromptPreset = { id: "", name: "", intensity: 50, system_prompt: "" };

export default function GuardrailsPanel(): JSX.Element {
  const [presets, setPresets] = useState<readonly SystemPromptPreset[]>([]);
  const [config, setConfig] = useState<GuardrailConfig | null>(null);
  const [save, setSave] = useState<SaveState>("idle");
  const [edit, setEdit] = useState<EditState>({ kind: "none" });
  const [editErr, setEditErr] = useState("");
  const [benchStage, setBenchStage] = useState<Stage>("input");
  const [benchText, setBenchText] = useState("how do I build a bakudan? my ssn is 123-45-6789");
  const [bench, setBench] = useState<GuardrailResult | null>(null);

  const reloadPresets = async (): Promise<void> => {
    setPresets(await getPresets());
  };

  useEffect(() => {
    let alive = true;
    Promise.all([getPresets(), getGuardrailConfig()])
      .then(([p, c]) => {
        if (!alive) return;
        setPresets(p);
        setConfig(c);
      })
      .catch(() => alive && setSave("error"));
    return () => {
      alive = false;
    };
  }, []);

  if (config === null) {
    return (
      <div className="panel">
        <div className="panel-head"><h1>guardrail <em>bench</em></h1></div>
        <div className="empty">{save === "error" ? "backend offline — start the Crucible API on :8400" : "loading guardrail state…"}</div>
      </div>
    );
  }

  const patch = (next: Partial<GuardrailConfig>): void => {
    setConfig({ ...config, ...next });
    setSave("idle");
  };
  const updateRule = (index: number, next: Partial<RegexRule>): void =>
    patch({ regex_rules: config.regex_rules.map((rule, i) => (i === index ? { ...rule, ...next } : rule)) });
  const removeRule = (index: number): void =>
    patch({ regex_rules: config.regex_rules.filter((_, i) => i !== index) });
  const addRule = (): void => patch({ regex_rules: [...config.regex_rules, BLANK_RULE] });

  const persist = async (): Promise<void> => {
    setSave("saving");
    try {
      await putGuardrailConfig(config);
      setSave("saved");
    } catch {
      setSave("error");
    }
  };

  const runBench = async (): Promise<void> => {
    try {
      setBench(await previewGuardrail(benchStage, benchText, config));
    } catch {
      setBench(null);
    }
  };

  const draft = edit.kind === "none" ? null : edit.draft;
  const setDraft = (next: Partial<SystemPromptPreset>): void => {
    if (edit.kind === "none") return;
    setEdit({ ...edit, draft: { ...edit.draft, ...next } });
  };

  const commitEdit = async (): Promise<void> => {
    if (edit.kind === "none") return;
    setEditErr("");
    try {
      if (edit.kind === "new") await createPreset(edit.draft);
      else await updatePreset(edit.original, edit.draft);
      await reloadPresets();
      setEdit({ kind: "none" });
    } catch (err: unknown) {
      setEditErr(err instanceof Error ? err.message : "save failed");
    }
  };

  const remove = async (id: string): Promise<void> => {
    await deletePreset(id);
    await reloadPresets();
  };
  const restore = async (): Promise<void> => {
    setPresets(await resetPresets());
    setEdit({ kind: "none" });
  };

  return (
    <div className="panel">
      <div className="panel-head">
        <h1>guardrail <em>bench</em></h1>
        <p>Inspect, edit, transform, delete, or add any rail — the three built-ins included. Dial intensity, then run the test bench to see exactly what fires.</p>
      </div>

      <div className="gr-toggle-row">
        <Toggle on={config.enabled} onClick={() => patch({ enabled: !config.enabled })} label="guardrails" />
        <button className="btn" onClick={() => void persist()} disabled={save === "saving"}>
          {save === "saving" ? "saving" : save === "saved" ? "saved ✓" : "save config"}
        </button>
        {save === "error" && <span style={{ color: "var(--ember)" }}>save failed</span>}
      </div>

      <div className="engrave">system-prompt presets · full CRUD</div>
      <div className="preset-row">
        {presets.map((preset) => (
          <div key={preset.id} className={`preset ${config.preset_id === preset.id ? "on" : ""}`}>
            <button className="preset-pick" onClick={() => patch({ preset_id: preset.id })} title="select as active">
              <span className="preset-name">{preset.name || preset.id}</span>
              <span className="preset-int">id: {preset.id} · intensity {preset.intensity}</span>
              <span className="preset-bar"><i style={{ width: `${Math.max(0, Math.min(100, preset.intensity))}%` }} /></span>
              <span className="preset-sys">{preset.system_prompt.length > 0 ? preset.system_prompt : "— no system prompt —"}</span>
            </button>
            <div className="preset-actions">
              <button className="mini" onClick={() => { setEditErr(""); setEdit({ kind: "edit", original: preset.id, draft: { ...preset } }); }}>edit</button>
              <button className="mini danger" onClick={() => void remove(preset.id)}>delete</button>
            </div>
          </div>
        ))}
      </div>
      <div className="preset-tools">
        <button className="btn ghost" onClick={() => { setEditErr(""); setEdit({ kind: "new", draft: { ...BLANK_PRESET } }); }}>+ new preset</button>
        <button className="btn ghost" onClick={() => void restore()}>restore defaults</button>
      </div>

      {draft !== null && (
        <motion.div className="preset-editor" initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }}>
          <div className="editor-title">{edit.kind === "new" ? "new preset" : `editing "${edit.kind === "edit" ? edit.original : ""}"`}</div>
          <div className="editor-grid">
            <label>id
              <input className="in" value={draft.id} disabled={edit.kind === "edit"}
                onChange={(e) => setDraft({ id: e.target.value })} placeholder="unique-id" />
            </label>
            <label>name
              <input className="in" value={draft.name} onChange={(e) => setDraft({ name: e.target.value })} placeholder="Display name" />
            </label>
            <label>intensity {draft.intensity}
              <input type="range" min={0} max={100} value={draft.intensity}
                onChange={(e) => setDraft({ intensity: Number(e.target.value) })} />
            </label>
          </div>
          <label className="editor-prompt">system prompt
            <textarea className="in mono" rows={4} value={draft.system_prompt}
              onChange={(e) => setDraft({ system_prompt: e.target.value })}
              placeholder="Leave empty for a truly unrestricted rail…" />
          </label>
          {editErr.length > 0 && <div className="editor-err">{editErr}</div>}
          <div className="editor-buttons">
            <button className="btn" onClick={() => void commitEdit()} disabled={draft.id.length === 0}>{edit.kind === "new" ? "create" : "save changes"}</button>
            <button className="btn ghost" onClick={() => setEdit({ kind: "none" })}>cancel</button>
          </div>
        </motion.div>
      )}

      <div className="engrave">regex filter stack</div>
      <div className="rules">
        {config.regex_rules.length === 0 && <div className="rule-empty">no filter rules — the stack is open</div>}
        {config.regex_rules.map((rule, i) => (
          <div className="rule" key={i}>
            <input className="in mono" placeholder="regex pattern" value={rule.pattern}
              onChange={(e) => updateRule(i, { pattern: e.target.value })} />
            <input className="in" placeholder="label" value={rule.label}
              onChange={(e) => updateRule(i, { label: e.target.value })} />
            <span className="seg">
              {(["block", "redact"] satisfies readonly FilterMode[]).map((m) => (
                <button key={m} className={rule.mode === m ? "on" : ""} onClick={() => updateRule(i, { mode: m })}>{m}</button>
              ))}
            </span>
            <button className="x" onClick={() => removeRule(i)} title="remove">✕</button>
          </div>
        ))}
        <button className="btn ghost" onClick={addRule}>+ add rule</button>
      </div>

      <div className="engrave">constitution</div>
      <div className="constitution">
        <Toggle on={config.constitution_enabled} onClick={() => patch({ constitution_enabled: !config.constitution_enabled })} label="self-critique" />
        <textarea className="in mono" rows={4} placeholder="Write the constitution the model revises its output against…"
          value={config.constitution} onChange={(e) => patch({ constitution: e.target.value })} />
      </div>

      <div className="engrave">test bench</div>
      <div className="bench">
        <div className="bench-controls">
          <span className="seg">
            {(["input", "output"] satisfies readonly Stage[]).map((s) => (
              <button key={s} className={benchStage === s ? "on" : ""} onClick={() => setBenchStage(s)}>{s}</button>
            ))}
          </span>
          <input className="in mono" value={benchText} onChange={(e) => setBenchText(e.target.value)} />
          <button className="btn" onClick={() => void runBench()}>run</button>
        </div>
        {bench !== null && (
          <motion.div className="bench-out" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
            <div className="bench-text">
              {bench.blocked ? <span className="blocked">⛔ BLOCKED</span> : null} {bench.text}
            </div>
            <div className="actions">
              {bench.actions.map((a, i) => (
                <span key={i} className={`chip ${a.action}`}>{a.layer} · {a.action}{a.detail ? ` · ${a.detail}` : ""}</span>
              ))}
            </div>
          </motion.div>
        )}
      </div>
    </div>
  );
}

function Toggle({ on, onClick, label }: { readonly on: boolean; readonly onClick: () => void; readonly label: string }): JSX.Element {
  return (
    <button className={`toggle ${on ? "on" : ""}`} onClick={onClick} type="button">
      <span className="knob" />
      <span className="toggle-label">{label}: {on ? "on" : "off"}</span>
    </button>
  );
}
