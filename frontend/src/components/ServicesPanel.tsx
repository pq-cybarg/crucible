import type { JSX } from "react";
import { useState } from "react";
import { motion } from "framer-motion";
import {
  detectServices,
  getActiveChatService,
  setActiveChatService,
} from "../services";
import type { DetectedService } from "../services";

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

  async function runScan(): Promise<void> {
    setScan({ state: "scanning" });
    const extra = custom.trim().length > 0 ? [custom.trim()] : [];
    const found = await detectServices(extra);
    setScan({ state: "done", found });
  }

  function pick(svc: DetectedService | null): void {
    setActiveChatService(svc);
    setActive(svc);
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
            chat → <b>{active.name}</b> <code>{active.baseUrl}</code>
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
                {s.models.length > 0 && (
                  <div className="byo-models">
                    {s.models.slice(0, 4).map((m) => (
                      <span key={m} className="byo-model">{m}</span>
                    ))}
                    {s.models.length > 4 && <span className="byo-model more">+{s.models.length - 4}</span>}
                  </div>
                )}
                {s.chat && (
                  <button
                    className={`btn byo-use ${isActive ? "on" : ""}`}
                    onClick={() => pick(isActive ? null : s)}
                  >
                    {isActive ? "active for chat ✓" : "use for chat"}
                  </button>
                )}
              </motion.div>
            );
          })}
        </div>
      )}
    </div>
  );
}
