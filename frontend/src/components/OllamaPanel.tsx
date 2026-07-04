import type { JSX } from "react";
import { useState } from "react";
import { motion } from "framer-motion";
import { getApiBase, getApiToken } from "../api";

// Self-contained (no api.ts edits): import Ollama's downloaded GGUF blobs as first-class
// Crucible models — then they're uncensorable, editable, quantizable, retrainable, servable.
interface OllamaModel {
  readonly name: string;
  readonly gguf_path: string;
  readonly exists: boolean;
  readonly size: number;
  readonly suggested_id: string;
  readonly imported: boolean;
}

function auth(): Record<string, string> {
  const t = getApiToken();
  return t ? { Authorization: `Bearer ${t}` } : {};
}

export default function OllamaPanel(): JSX.Element {
  const [rows, setRows] = useState<readonly OllamaModel[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [importing, setImporting] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);

  async function scan(): Promise<void> {
    setBusy(true); setErr(null);
    try {
      const r = await fetch(getApiBase() + "/api/models/ollama", { headers: auth() });
      if (!r.ok) throw new Error(`scan ${r.status}`);
      setRows((await r.json()) as OllamaModel[]);
    } catch (e: unknown) {
      setErr(e instanceof Error ? e.message : "scan failed (is Crucible running?)");
    } finally { setBusy(false); }
  }

  async function doImport(name: string): Promise<void> {
    setImporting(name); setErr(null);
    try {
      const r = await fetch(getApiBase() + "/api/models/import-ollama", {
        method: "POST", headers: { "Content-Type": "application/json", ...auth() },
        body: JSON.stringify({ name }),
      });
      if (!r.ok) throw new Error(((await r.json().catch(() => ({}))) as { detail?: string }).detail ?? `import ${r.status}`);
      setRows((prev) => prev?.map((m) => (m.name === name ? { ...m, imported: true } : m)) ?? prev);
    } catch (e: unknown) {
      setErr(e instanceof Error ? e.message : "import failed");
    } finally { setImporting(null); }
  }

  return (
    <div className="byo">
      <div className="byo-head">
        <h2>Ollama <em>· import your pulled models</em></h2>
        <p>
          Grab the raws: import Ollama's downloaded GGUF blobs as first-class Crucible models —
          then they're yours to <b>uncensor</b>, edit, quantize, <b>retrain</b>, and serve
          (llama.cpp), and they join the runtime round-robin.
        </p>
      </div>
      <div className="byo-controls">
        <button className="btn" onClick={() => void scan()} disabled={busy}>
          {busy ? "scanning…" : "scan Ollama"}
        </button>
        {err && <span className="runtime-err">{err}</span>}
      </div>
      {rows !== null && rows.length === 0 && (
        <div className="empty">no Ollama models found (set <code>OLLAMA_MODELS</code> if non-default)</div>
      )}
      {rows !== null && rows.length > 0 && (
        <div className="byo-grid">
          {rows.map((m, i) => (
            <motion.div key={m.name} className={`byo-card ${m.imported ? "on" : ""}`}
              initial={{ opacity: 0, y: 4 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.03 }}>
              <div className="byo-card-top">
                <span className="byo-name" title={m.name}>{m.name}</span>
                <span className="byo-model">{(m.size / 1e9).toFixed(1)} GB</span>
              </div>
              <code className="byo-url">{m.suggested_id}</code>
              <button className={`btn byo-use ${m.imported ? "on" : ""}`} disabled={importing === m.name || !m.exists}
                onClick={() => void doImport(m.name)}>
                {m.imported ? "imported ✓" : importing === m.name ? "importing…" : "import"}
              </button>
            </motion.div>
          ))}
        </div>
      )}
    </div>
  );
}
