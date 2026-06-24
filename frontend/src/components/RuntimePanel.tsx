import type { JSX } from "react";
import { useCallback, useEffect, useState } from "react";
import { motion } from "framer-motion";
import { getRuntime, setActiveModels, startModel, stopModel } from "../api";
import type { ModelRow, RuntimeStatus } from "../api";

function isLaunchable(row: ModelRow): boolean {
  return row.path.endsWith(".gguf") && !row.endpoint;
}

export default function RuntimePanel({ rows }: { readonly rows: readonly ModelRow[] }): JSX.Element {
  const [status, setStatus] = useState<RuntimeStatus | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const refresh = useCallback(async (): Promise<void> => {
    try { setStatus(await getRuntime()); } catch { /* backend offline */ }
  }, []);

  useEffect(() => {
    void refresh();
    const h = window.setInterval(() => void refresh(), 4000);
    return () => window.clearInterval(h);
  }, [refresh]);

  const launchable = rows.filter(isLaunchable);
  const resident = new Map((status?.resident ?? []).map((r) => [r.model_id, r]));
  const activeSet = new Set(status?.active ?? []);

  async function toggleRun(id: string): Promise<void> {
    setBusy(id); setErr(null);
    try {
      if (resident.has(id)) setStatus(await stopModel(id));
      else { const r = await startModel(id); setStatus(r.status); }
    } catch (e: unknown) {
      setErr(e instanceof Error ? e.message : "runtime error");
    } finally { setBusy(null); void refresh(); }
  }

  async function toggleActive(id: string): Promise<void> {
    const next = new Set(activeSet);
    if (next.has(id)) next.delete(id); else next.add(id);
    try { setStatus(await setActiveModels([...next])); } catch { /* offline */ }
  }

  if (launchable.length === 0) return <></>;

  return (
    <div className="runtime">
      <div className="runtime-head">
        <h2>runtime <em>· load models on demand</em></h2>
        <p>
          Launch a local GGUF model's server right here. Keeps up to <b>{status?.max_resident ?? 1}</b>{" "}
          model(s) in memory at once; mark several <b>active</b> and — when memory is tight — they
          <b> round-robin</b> (the least-recently-used one is unloaded to make room). Raise the cap with{" "}
          <code>CRUCIBLE_MAX_RESIDENT</code>.
        </p>
      </div>
      {err && <div className="runtime-err">{err}</div>}
      <div className="runtime-grid">
        {launchable.map((row) => {
          const inst = resident.get(row.id);
          const running = inst !== undefined;
          return (
            <motion.div key={row.id} className={`runtime-card ${running ? "on" : ""}`}
              initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }}>
              <div className="runtime-top">
                <span className="runtime-name">{row.name}</span>
                <span className={`runtime-dot ${running ? "live" : "cold"}`} />
              </div>
              <code className="runtime-sub">{running ? inst.endpoint : row.id}</code>
              <div className="runtime-actions">
                <button className={`btn runtime-run ${running ? "stop" : ""}`}
                  disabled={busy === row.id} onClick={() => void toggleRun(row.id)}>
                  {busy === row.id ? "…" : running ? "stop" : "start"}
                </button>
                <label className="runtime-active" title="include in the round-robin set">
                  <input type="checkbox" checked={activeSet.has(row.id)} onChange={() => void toggleActive(row.id)} />
                  active
                </label>
              </div>
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}
