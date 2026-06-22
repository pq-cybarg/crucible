import type { JSX } from "react";
import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { getModels } from "../api";
import type { ModelRow } from "../api";
import ServicesPanel from "./ServicesPanel";

type Load =
  | { readonly state: "loading" }
  | { readonly state: "error"; readonly message: string }
  | { readonly state: "ready"; readonly rows: readonly ModelRow[] };

export default function ModelsPanel(): JSX.Element {
  const [load, setLoad] = useState<Load>({ state: "loading" });

  useEffect(() => {
    let alive = true;
    getModels()
      .then((rows) => alive && setLoad({ state: "ready", rows }))
      .catch((err: unknown) =>
        alive && setLoad({ state: "error", message: err instanceof Error ? err.message : "unknown error" }),
      );
    return () => {
      alive = false;
    };
  }, []);

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
        <table className="grid-table">
          <thead>
            <tr>
              <th>id</th><th>name</th><th>kind</th><th>quant</th><th>endpoint</th><th>created</th>
            </tr>
          </thead>
          <tbody>
            {load.rows.map((row, i) => (
              <motion.tr
                key={row.id}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: i * 0.04 }}
              >
                <td style={{ color: "var(--bone)" }}>{row.id}</td>
                <td>{row.name}</td>
                <td><span className={`kind ${row.kind}`}>{row.kind}</span></td>
                <td>{row.quant}</td>
                <td>{row.endpoint ?? "—"}</td>
                <td style={{ color: "var(--ash)" }}>{row.created}</td>
              </motion.tr>
            ))}
          </tbody>
        </table>
      )}

      <ServicesPanel />
    </div>
  );
}
