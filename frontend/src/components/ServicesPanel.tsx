import type { JSX } from "react";
import { useState } from "react";
import { motion } from "framer-motion";
import {
  connectService,
  detectServices,
  getActiveChatService,
  getChatMode,
  setActiveChatService,
} from "../services";
import type { ChatMode, DetectedService } from "../services";
import { getApiBase, getApiToken, getHealth } from "../api";

type Scan =
  | { readonly state: "idle" }
  | { readonly state: "scanning" }
  | { readonly state: "done"; readonly found: readonly DetectedService[] };

function svcKey(s: DetectedService): string {
  return `${s.type}@${s.baseUrl}`;
}

export default function ServicesPanel(): JSX.Element {
  const [scan, setScan] = useState<Scan>({ state: "idle" });
  const [custom, setCustom] = useState("");
  const [active, setActive] = useState<DetectedService | null>(getActiveChatService());
  const [mode, setMode] = useState<ChatMode>(getChatMode());
  const [busyKey, setBusyKey] = useState<string | null>(null);
  // per-card chosen model (when a service exposes several); keyed by svcKey
  const [picked, setPicked] = useState<Readonly<Record<string, string>>>({});

  function modelFor(svc: DetectedService): string | undefined {
    return picked[svcKey(svc)] ?? svc.models[0];
  }

  async function runScan(): Promise<void> {
    setScan({ state: "scanning" });
    const extra = custom.trim().length > 0 ? [custom.trim()] : [];
    const found = await detectServices(extra);
    setScan({ state: "done", found });
  }

  function pick(svc: DetectedService | null, m: ChatMode = "direct"): void {
    setActiveChatService(svc, m, svc ? modelFor(svc) : undefined);
    setActive(svc);
    setMode(m);
  }

  // "drive with tools": register the endpoint with a reachable Crucible backend, then
  // route the forge through it (full tool-loop). Needs a Crucible node online.
  async function driveWithTools(svc: DetectedService): Promise<void> {
    setBusyKey(svcKey(svc));
    try {
      if (!(await getHealth())) {
        setScan((s) => s);
        alert(
          "Tool-loop needs a Crucible backend online. Start Crucible (or set the node URL " +
            "top-right) — it runs the tools and relays generation to this service.",
        );
        return;
      }
      await connectService(getApiBase(), svc, getApiToken());
      pick(svc, "tools");
    } catch (err: unknown) {
      alert(`could not connect via Crucible: ${err instanceof Error ? err.message : "failed"}`);
    } finally {
      setBusyKey(null);
    }
  }

  const activeKey = active ? svcKey(active) : null;

  return (
    <div className="byo">
      <div className="byo-head">
        <h2>BYO-AI <em>· connect your own backends</em></h2>
        <p>
          Scan localhost (and any remote you name) for Crucible, Ollama, llama.cpp, vLLM, or ComfyUI.
          Chat-capable services drive the <b>forge</b> directly from this page — even the static demo.
          To <b>abliterate / edit</b> weights you still need a Crucible node with <b>write access</b> to
          the model: run Crucible locally (it can wrap any of these), or point it at a remote you can write to.
        </p>
      </div>

      <div className="byo-controls">
        <input
          className="node-input"
          style={{ width: 260 }}
          value={custom}
          onChange={(e) => setCustom(e.target.value)}
          placeholder="extra remote, e.g. http://gpu-node:8081"
          title="any OpenAI-compatible /v1 endpoint — added to the scan"
        />
        <button className="btn" onClick={() => void runScan()} disabled={scan.state === "scanning"}>
          {scan.state === "scanning" ? "scanning…" : "scan"}
        </button>
        {active && (
          <span className="byo-active">
            {mode === "tools" ? "forge+tools" : "chat"} → <b>{active.name}</b>{" "}
            <code>{active.baseUrl}</code>
            <button className="byo-clear" onClick={() => pick(null)} title="back to Crucible's own agent">
              use Crucible
            </button>
          </span>
        )}
      </div>

      {scan.state === "done" && scan.found.length === 0 && (
        <div className="empty">
          <div className="big">nothing answered</div>
          No service responded on the probed ports. Start Ollama (<code>:11434</code>),
          llama.cpp <code>--port 8080</code>, or a Crucible node — then scan again. For cross-origin
          access from a browser, Ollama needs <code>OLLAMA_ORIGINS=*</code> (or this page's origin).
        </div>
      )}

      {scan.state === "done" && scan.found.length > 0 && (
        <div className="byo-grid">
          {scan.found.map((s, i) => {
            const isActive = activeKey === svcKey(s);
            return (
              <motion.div
                key={svcKey(s)}
                className={`byo-card ${isActive ? "on" : ""}`}
                initial={{ opacity: 0, y: 6 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.05 }}
              >
                <div className="byo-card-top">
                  <span className="byo-name">{s.name}</span>
                  <div className="byo-badges">
                    {s.full && <span className="byo-badge full">full</span>}
                    {s.chat && !s.full && <span className="byo-badge chat">chat</span>}
                    {!s.chat && <span className="byo-badge cold">no chat</span>}
                  </div>
                </div>
                <code className="byo-url">{s.baseUrl}</code>
                <div className="byo-note">{s.note}</div>
                {s.models.length === 1 && (
                  <div className="byo-models">
                    <span className="byo-model">{s.models[0]}</span>
                  </div>
                )}
                {s.models.length > 1 && (
                  <select
                    className="byo-modelsel"
                    value={modelFor(s)}
                    title="choose which model this service should serve"
                    onChange={(e) => {
                      const m = e.target.value;
                      setPicked((p) => ({ ...p, [svcKey(s)]: m }));
                      // keep the active selection's model in sync if this card is active
                      if (isActive) setActiveChatService(s, mode, m);
                    }}
                  >
                    {s.models.map((m) => (
                      <option key={m} value={m}>{m}</option>
                    ))}
                  </select>
                )}
                {s.chat && (
                  <div className="byo-actions">
                    <button
                      className={`btn byo-use ${isActive && mode === "direct" ? "on" : ""}`}
                      onClick={() => pick(isActive && mode === "direct" ? null : s, "direct")}
                      title="plain chat from the browser — works on the static page; no tools"
                    >
                      {isActive && mode === "direct" ? "chat ✓" : "use for chat"}
                    </button>
                    {!s.full && (
                      <button
                        className={`btn byo-use ${isActive && mode === "tools" ? "on" : ""}`}
                        disabled={busyKey === svcKey(s)}
                        onClick={() =>
                          isActive && mode === "tools" ? pick(null) : void driveWithTools(s)
                        }
                        title="drive with the full Crucible agent tool-loop (needs a Crucible backend online)"
                      >
                        {busyKey === svcKey(s)
                          ? "connecting…"
                          : isActive && mode === "tools"
                            ? "tools ✓"
                            : "+ tools (via Crucible)"}
                      </button>
                    )}
                  </div>
                )}
              </motion.div>
            );
          })}
        </div>
      )}
    </div>
  );
}
