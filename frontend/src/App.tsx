import { useEffect, useState } from "react";
import type { JSX } from "react";
import { AnimatePresence, motion } from "framer-motion";
import AgentConsole from "./components/AgentConsole";
import ModelsPanel from "./components/ModelsPanel";
import GuardrailsPanel from "./components/GuardrailsPanel";
import UncensorPanel from "./components/UncensorPanel";
import { BenchmarksPanel, WeightsPanel } from "./components/Panels";
import { getHealth, getModels } from "./api";

type TabId = "agent" | "models" | "guardrails" | "uncensor" | "weights" | "benchmarks";

interface Tab {
  readonly id: TabId;
  readonly label: string;
  readonly path: string;
}

const TABS: readonly Tab[] = [
  { id: "agent", label: "forge", path: "M4 13h7l-1 7 10-11h-7l1-7z" },
  { id: "models", label: "models", path: "M4 7l8-4 8 4-8 4-8-4zm0 5l8 4 8-4m-16 5l8 4 8-4" },
  { id: "guardrails", label: "guards", path: "M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6l8-3z" },
  { id: "uncensor", label: "ablit", path: "M5 19l7-14 7 14M8 14h8" },
  { id: "weights", label: "weights", path: "M4 6h16M4 12h16M4 18h10M18 16v4M16 18h4" },
  { id: "benchmarks", label: "bench", path: "M5 20V10M10 20V4M15 20v-8M20 20V7" },
];

function Panel({ tab }: { readonly tab: TabId }): JSX.Element {
  switch (tab) {
    case "agent":
      return <AgentConsole />;
    case "models":
      return <ModelsPanel />;
    case "guardrails":
      return <GuardrailsPanel />;
    case "uncensor":
      return <UncensorPanel />;
    case "weights":
      return <WeightsPanel />;
    case "benchmarks":
      return <BenchmarksPanel />;
  }
}

export default function App(): JSX.Element {
  const [tab, setTab] = useState<TabId>("agent");
  const [online, setOnline] = useState(false);
  const [modelCount, setModelCount] = useState<number | null>(null);

  useEffect(() => {
    let alive = true;
    const poll = async (): Promise<void> => {
      const ok = await getHealth();
      if (!alive) return;
      setOnline(ok);
      if (ok) {
        try {
          const rows = await getModels();
          if (alive) setModelCount(rows.length);
        } catch {
          if (alive) setModelCount(null);
        }
      } else if (alive) {
        setModelCount(null);
      }
    };
    void poll();
    const handle = window.setInterval(() => void poll(), 4000);
    return () => {
      alive = false;
      window.clearInterval(handle);
    };
  }, []);

  const hot = modelCount !== null && modelCount > 0;

  return (
    <div className="app">
      <nav className="rail">
        <div className="brand"><div className="mark" /></div>
        {TABS.map((t, i) => (
          <motion.button
            key={t.id}
            type="button"
            className={`tab ${tab === t.id ? "active" : ""}`}
            onClick={() => setTab(t.id)}
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.05 * i + 0.1 }}
          >
            <svg className="glyph" viewBox="0 0 24 24" strokeLinecap="round" strokeLinejoin="round">
              <path d={t.path} />
            </svg>
            <span className="lbl">{t.label}</span>
          </motion.button>
        ))}
      </nav>

      <header className="bar">
        <span className="crumb">
          crucible<span className="sub"> · {tab}</span>
        </span>
        <div className="telemetry">
          <span className="stat">
            <span className={`dot ${online ? "ok" : "cold"}`} />
            api <b>{online ? "online" : "offline"}</b>
          </span>
          <span className="stat">
            <span className={`dot ${hot ? "hot" : "cold"}`} />
            inference <b>{hot ? "loaded" : "cold"}</b>
          </span>
          <span className="stat">
            models <b>{modelCount ?? "—"}</b>
          </span>
          <span className="stat">node <b>mac · m5</b></span>
        </div>
      </header>

      <main className="main">
        <AnimatePresence mode="wait">
          <motion.div
            key={tab}
            style={{ height: "100%" }}
            initial={{ opacity: 0, y: 6 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -6 }}
            transition={{ duration: 0.18 }}
          >
            <Panel tab={tab} />
          </motion.div>
        </AnimatePresence>
      </main>
    </div>
  );
}
