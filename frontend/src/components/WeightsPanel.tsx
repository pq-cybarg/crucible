import { useEffect, useMemo, useState } from "react";
import type { JSX } from "react";
import { motion } from "framer-motion";
import { getModels, getWeights } from "../api";
import type { ModelRow, TensorInfo, WeightsResult } from "../api";

const DTYPE_COLORS: Record<string, string> = {
  F32: "#aeb6c1", F16: "#8a9099", BF16: "#767d87",
  Q8_0: "#ff8c3f", Q6_K: "#ff6a1a", Q5_K: "#ff5a14", Q5_0: "#e0560f",
  Q4_K: "#2fd1a6", Q4_0: "#28b892", Q3_K: "#ff3b2f", Q2_K: "#c43025",
};

function human(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return String(n);
}

function layerOf(name: string): number {
  const m = /blk\.(\d+)\./.exec(name);
  return m && m[1] !== undefined ? Number(m[1]) : -1;
}

export default function WeightsPanel(): JSX.Element {
  const [models, setModels] = useState<readonly ModelRow[]>([]);
  const [modelId, setModelId] = useState("");
  const [res, setRes] = useState<WeightsResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [layerFilter, setLayerFilter] = useState<string>("all");
  const [search, setSearch] = useState("");

  useEffect(() => {
    let alive = true;
    getModels().then((m) => {
      if (!alive) return;
      setModels(m);
      const first = m[0];
      if (first) setModelId(first.id);
    }).catch(() => undefined);
    return () => { alive = false; };
  }, []);

  const view = res !== null && res.kind === "view" ? res.view : null;

  const layers = useMemo(() => {
    if (view === null) return [];
    const set = new Set<number>();
    for (const t of view.tensors) set.add(layerOf(t.name));
    return [...set].sort((a, b) => a - b);
  }, [view]);

  const visible: readonly TensorInfo[] = useMemo(() => {
    if (view === null) return [];
    return view.tensors.filter((t) => {
      const okLayer = layerFilter === "all" || String(layerOf(t.name)) === layerFilter;
      const okSearch = search === "" || t.name.toLowerCase().includes(search.toLowerCase());
      return okLayer && okSearch;
    });
  }, [view, layerFilter, search]);

  const inspect = async (): Promise<void> => {
    if (modelId === "") return;
    setBusy(true);
    setRes(null);
    setRes(await getWeights(modelId));
    setBusy(false);
  };

  return (
    <div className="panel">
      <div className="panel-head">
        <h1>weight <em>explorer</em></h1>
        <p>See what a model is actually made of — explained in plain language first (what it is, how it
          thinks, where behaviors live and how to change them), with the exact technical structure below
          for when you need it. Nothing is loaded into memory; this just reads the map.</p>
      </div>

      <div className="abl-controls">
        <label className="fld">model
          <select className="in" value={modelId} onChange={(e) => setModelId(e.target.value)}>
            {models.length === 0 && <option value="">— no models registered —</option>}
            {models.map((m) => <option key={m.id} value={m.id}>{m.name} · {m.quant}</option>)}
          </select>
        </label>
        <button className="btn" onClick={() => void inspect()} disabled={busy || modelId === ""}>{busy ? "reading…" : "inspect weights"}</button>
      </div>

      {res !== null && res.kind === "no-file" && <div className="abl-note">model file isn't on this machine's disk yet — inspect runs against the local GGUF once it's downloaded.</div>}
      {res !== null && res.kind === "offline" && <div className="abl-note err">backend offline.</div>}

      {view !== null && (
        <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}>
          {view.explain && (
            <>
              <div className="plain-card">
                <div className="plain-headline">{view.explain.model.headline}</div>
                <p className="plain-line"><span className="plain-tag">what it is</span>{view.explain.model.what_it_is}</p>
                <p className="plain-line"><span className="plain-tag">how it works</span>{view.explain.model.how_it_works}</p>
                <p className="plain-line"><span className="plain-tag">its size</span>{view.explain.model.size_meaning}</p>
                <p className="plain-line"><span className="plain-tag">how to change it</span>{view.explain.model.how_to_change}</p>
              </div>

              <div className="engrave">the layer journey · early → late</div>
              <div className="layer-journey">
                {view.explain.layers.map((l) => (
                  <div key={l.layer} className={`journey-seg band-${l.band}`}
                    title={`layer ${l.layer} (${l.band}): ${l.role}`} />
                ))}
              </div>
              <div className="journey-legend">
                {Object.entries(view.explain.legend).map(([band, role]) => (
                  <span key={band} className="journey-key">
                    <i className={`band-dot band-${band}`} /><b>{band}</b> — {role}
                  </span>
                ))}
              </div>
            </>
          )}

          <div className="engrave">by the numbers</div>
          <div className="w-summary">
            <div className="w-stat"><div className="v">{view.summary.architecture ?? "—"}</div><div className="k">architecture</div></div>
            <div className="w-stat"><div className="v">{human(view.summary.total_params)}</div><div className="k">parameters</div></div>
            <div className="w-stat"><div className="v">{view.summary.n_layers}</div><div className="k">layers</div></div>
            <div className="w-stat"><div className="v">{view.summary.n_tensors}</div><div className="k">tensors</div></div>
          </div>

          <div className="engrave">quantization mix</div>
          <div className="dtype-bar">
            {Object.entries(view.summary.dtypes).sort((a, b) => b[1] - a[1]).map(([dt, count]) => (
              <div key={dt} className="dtype-seg"
                style={{ width: `${(count / view.summary.n_tensors) * 100}%`, background: DTYPE_COLORS[dt] ?? "#3a4150" }}
                title={`${dt}: ${count} tensors`}>
                {(count / view.summary.n_tensors) > 0.08 ? dt : ""}
              </div>
            ))}
          </div>

          <div className="engrave">tensors</div>
          <div className="w-filter">
            <label className="fld">layer
              <select className="in" value={layerFilter} onChange={(e) => setLayerFilter(e.target.value)}>
                <option value="all">all</option>
                {layers.map((l) => <option key={l} value={String(l)}>{l < 0 ? "non-block" : `blk.${l}`}</option>)}
              </select>
            </label>
            <label className="fld">filter
              <input className="in" value={search} onChange={(e) => setSearch(e.target.value)} placeholder="name contains…" />
            </label>
            <span style={{ color: "var(--ash)", fontSize: 11, paddingBottom: 9 }}>{visible.length} shown</span>
          </div>
          <table className="grid-table">
            <thead><tr><th>tensor</th><th>shape</th><th>dtype</th><th>params</th></tr></thead>
            <tbody>
              {visible.slice(0, 400).map((t) => (
                <tr key={t.name}>
                  <td style={{ color: "var(--bone)" }}>{t.name}</td>
                  <td>[{t.shape.join(" × ")}]</td>
                  <td><span style={{ color: DTYPE_COLORS[t.dtype] ?? "var(--steel-text)" }}>{t.dtype}</span></td>
                  <td style={{ color: "var(--ash)" }}>{human(t.n_params)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {visible.length > 400 && <p style={{ color: "var(--ash)", fontSize: 11, marginTop: 8 }}>showing first 400 of {visible.length} — narrow with the layer/name filter.</p>}
        </motion.div>
      )}
    </div>
  );
}
