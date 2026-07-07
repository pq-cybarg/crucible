import type { JSX } from "react";
import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { forgetModel, getModels, repointModel } from "../api";
import type { ModelRow } from "../api";
import ServicesPanel from "./ServicesPanel";
import RuntimePanel from "./RuntimePanel";
import OllamaPanel from "./OllamaPanel";
import { getActiveModelId, setActiveModelId } from "../services";
import { getApiBase } from "../api";

type Load =
  | { readonly state: "loading" }
  | { readonly state: "error"; readonly message: string }
  | { readonly state: "ready"; readonly rows: readonly ModelRow[] };

export default function ModelsPanel(): JSX.Element {
  const [load, setLoad] = useState<Load>({ state: "loading" });
  const [activeId, setActiveId] = useState<string | null>(getActiveModelId());
  const [status, setStatus] = useState<Readonly<Record<string, { online: boolean; launchable: boolean; servable: boolean }>>>({});

  useEffect(() => {
    let live = true;
    const poll = (): void => {
      void fetch(getApiBase() + "/api/models/status")
        .then((r) => (r.ok ? r.json() : []))
        .then((rows: Array<{ id: string; online: boolean; launchable: boolean; servable: boolean }>) => {
          if (!live) return;
          const map: Record<string, { online: boolean; launchable: boolean; servable: boolean }> = {};
          for (const x of rows) map[x.id] = { online: x.online, launchable: x.launchable, servable: x.servable };
          setStatus(map);
        }).catch(() => undefined);
    };
    poll();
    const h = window.setInterval(poll, 4000);
    return () => { live = false; window.clearInterval(h); };
  }, []);

  const selectModel = (id: string): void => {
    const next = activeId === id ? null : id;
    setActiveModelId(next);
    setActiveId(next);
  };

  const refresh = (): void => {
    getModels()
      .then((rows) => setLoad({ state: "ready", rows }))
      .catch((err: unknown) => setLoad({ state: "error", message: err instanceof Error ? err.message : "unknown error" }));
  };
  useEffect(() => { refresh(); }, []);

  const forget = (id: string): void => {
    if (!window.confirm(`Forget "${id}"? This removes the registry entry only — no weight files are deleted.`)) return;
    void forgetModel(id).then(() => { if (activeId === id) selectModel(id); refresh(); }).catch(() => undefined);
  };
  const repoint = (id: string): void => {
    const ep = window.prompt(`Re-point "${id}" at a live endpoint URL (e.g. your Ollama):`, "http://localhost:11434");
    if (ep == null || ep.trim().length === 0) return;
    void repointModel(id, ep.trim()).then(refresh).catch(() => undefined);
  };

  return (
    <div className="panel">
      <div className="panel-head">
        <h1>model <em>registry</em></h1>
        <p>Every base, abliterated, and steered variant — with immutable originals and full lineage.</p>
      </div>

      {load.state === "loading" && <div className="empty">reading registry…</div>}

      {load.state === "error" && (
        <div className="empty">
          <div className="big">registry unreachable</div>
          {load.message} — start the Crucible API on <code>:8400</code>
        </div>
      )}

      {load.state === "ready" && load.rows.length === 0 && (
        <div className="empty">
          <div className="big">the forge is cold</div>
          No models registered yet. The GLM-4-32B base lands here once its download completes.
        </div>
      )}

      {load.state === "ready" && load.rows.length > 0 && (
        <>
          <p className="hint">
            Click <b>use</b> to make the forge talk to that model. Models with an endpoint chat over
            the network; a local model with no endpoint uses the loaded HF adapter (needs the backend
            running with <code>CRUCIBLE_HF_MODEL</code>).
          </p>
          <table className="grid-table">
            <thead>
              <tr>
                <th>id</th><th>name</th><th>kind</th><th>quant</th><th>endpoint</th><th>created</th><th>chat</th>
              </tr>
            </thead>
            <tbody>
              {load.rows.map((row, i) => (
                <motion.tr
                  key={row.id}
                  className={activeId === row.id ? "row-active" : ""}
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: i * 0.04 }}
                >
                  <td style={{ color: "var(--bone)" }}>
                    <span className={`model-dot ${status[row.id]?.online ? "on" : status[row.id]?.launchable ? "warm" : status[row.id]?.servable ? "warm" : "off"}`}
                      title={status[row.id]?.online ? "online" : status[row.id]?.launchable ? "launchable" : status[row.id]?.servable ? "adapter-servable" : "offline / unservable"} />
                    {row.id}
                  </td>
                  <td>{row.name}</td>
                  <td><span className={`kind ${row.kind}`}>{row.kind}</span></td>
                  <td>{row.quant}</td>
                  <td>{row.endpoint ?? "—"}</td>
                  <td style={{ color: "var(--ash)" }}>{row.created}</td>
                  <td>
                    {status[row.id] && !status[row.id]?.servable ? (
                      <span className="model-fix" title="no live endpoint and not a launchable local GGUF — re-point it at a running server, or forget the entry">
                        <button className="btn row-use" onClick={() => repoint(row.id)}
                          title="aim this model at a live endpoint (e.g. your Ollama at :11434) to re-enable it">re-point</button>
                        <button className="btn ghost row-forget" onClick={() => forget(row.id)}
                          title="remove this dead registry entry (does not delete weight files)">forget</button>
                      </span>
                    ) : (
                      <button
                        className={`btn row-use ${activeId === row.id ? "on" : ""}`}
                        onClick={() => selectModel(row.id)}
                        title={status[row.id]?.launchable && !status[row.id]?.online
                          ? "no live endpoint — the forge will launch this local GGUF on first use"
                          : "route the forge console to this model"}
                      >
                        {activeId === row.id ? "active ✓" : "use"}
                      </button>
                    )}
                  </td>
                </motion.tr>
              ))}
            </tbody>
          </table>
        </>
      )}

      {load.state === "ready" && <RuntimePanel rows={load.rows} />}
      {load.state === "ready" && <OllamaPanel />}
      <ServicesPanel />
    </div>
  );
}
