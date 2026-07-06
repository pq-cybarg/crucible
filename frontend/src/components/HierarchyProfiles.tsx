import type { JSX } from "react";
import { useEffect, useState } from "react";
import { deleteProfile, getModels, getProfiles, saveProfile } from "../api";
import type { HierarchyLayer, HierarchyProfile, ModelRow } from "../api";

// Editor for agent-hierarchy profiles: for each LAYER of the spawn tree, pick the worker model
// (does the work) and its lighter COMMUNICATOR model (relays/compresses results up between layers).
// Multiple named profiles; deeper layers reuse the last one. Select a profile in the forge to use it.
type EditLayer = { worker: string; communicator: string };   // "" = default model
const toEdit = (l: HierarchyLayer): EditLayer => ({ worker: l.worker ?? "", communicator: l.communicator ?? "" });
const toApi = (l: EditLayer): HierarchyLayer => ({ worker: l.worker || null, communicator: l.communicator || null });

export default function HierarchyProfiles(): JSX.Element {
  const [profiles, setProfiles] = useState<readonly HierarchyProfile[]>([]);
  const [models, setModels] = useState<readonly ModelRow[]>([]);
  const [name, setName] = useState("");
  const [layers, setLayers] = useState<EditLayer[]>([{ worker: "", communicator: "" }]);
  const [note, setNote] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);

  async function refresh(): Promise<void> {
    const [p, m] = await Promise.all([getProfiles(), getModels().catch(() => [])]);
    setProfiles(p); setModels(m);
  }
  useEffect(() => { void refresh(); }, []);

  function edit(p: HierarchyProfile): void {
    setName(p.name);
    setLayers(p.layers.length ? p.layers.map(toEdit) : [{ worker: "", communicator: "" }]);
    setNote(null); setErr(null);
  }
  function fresh(): void { setName(""); setLayers([{ worker: "", communicator: "" }]); setNote(null); setErr(null); }

  const setLayer = (i: number, patch: Partial<EditLayer>): void =>
    setLayers((ls) => ls.map((l, j) => (j === i ? { ...l, ...patch } : l)));

  async function save(): Promise<void> {
    setErr(null); setNote(null);
    if (name.trim().length === 0) { setErr("name required"); return; }
    try {
      await saveProfile({ name: name.trim(), layers: layers.map(toApi) });
      setNote(`saved "${name.trim()}"`);
      await refresh();
    } catch (e: unknown) { setErr(e instanceof Error ? e.message : "save failed"); }
  }
  async function remove(n: string): Promise<void> {
    await deleteProfile(n);
    if (n === name) fresh();
    await refresh();
  }

  const Picker = ({ value, onChange, label }: { readonly value: string; readonly onChange: (v: string) => void; readonly label: string }): JSX.Element => (
    <label className="fld" style={{ flex: 1 }}>{label}
      <select className="in" value={value} onChange={(e) => onChange(e.target.value)}>
        <option value="">default model</option>
        {models.map((m) => <option key={m.id} value={m.id}>{m.name}</option>)}
      </select>
    </label>
  );

  return (
    <div className="hier-editor">
      <div className="engrave">agent hierarchy profiles · worker + communicator per layer</div>
      <p className="hint" style={{ marginTop: 0 }}>
        Each spawn-tree layer runs on a <b>worker</b> model; its lighter <b>communicator</b> (pick the
        weaker/simpler model) compresses a deep child's result before it climbs back up, so parents
        never process raw deep-leaf text. Deeper layers reuse the last one. Select a profile in the forge.
      </p>

      <div className="hier-list">
        {profiles.map((p) => (
          <div key={p.name} className={`hier-chip ${p.name === name ? "on" : ""}`}>
            <button className="hier-name" onClick={() => edit(p)}>{p.name}</button>
            <span className="hier-layers">{p.layers.length} layer{p.layers.length === 1 ? "" : "s"}</span>
            <button className="hier-del" title="delete" onClick={() => void remove(p.name)}>✕</button>
          </div>
        ))}
        <button className="btn ghost" onClick={fresh}>+ new</button>
      </div>

      <label className="fld">profile name
        <input className="in" value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. research, deep-dive" />
      </label>

      <div className="hier-layers-edit">
        {layers.map((l, i) => (
          <div className="hier-layer" key={i}>
            <span className="hier-depth">L{i}</span>
            <Picker value={l.worker} label="worker" onChange={(v) => setLayer(i, { worker: v })} />
            <Picker value={l.communicator} label="communicator (lighter)" onChange={(v) => setLayer(i, { communicator: v })} />
            {layers.length > 1 && <button className="hier-del" onClick={() => setLayers((ls) => ls.filter((_, j) => j !== i))}>✕</button>}
          </div>
        ))}
      </div>

      <div className="hier-actions">
        <button className="btn ghost" onClick={() => setLayers((ls) => [...ls, { worker: "", communicator: "" }])}>+ layer</button>
        <button className="btn" onClick={() => void save()}>save profile</button>
        {note && <span className="mem-note">{note}</span>}
        {err && <span className="runtime-err">{err}</span>}
      </div>
    </div>
  );
}
