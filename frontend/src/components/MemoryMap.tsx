import type { JSX } from "react";
import type { MemoryTreeNode } from "../api";

// Bubble map of crystallized memory. A tree of indented rows is precise but hard to feel; a bubble
// map shows the SHAPE of what's remembered — domains as big bubbles containing their sub-memories,
// each sized by how much it holds, so you see the structure of your knowledge at a glance. Chunked
// nodes nest their children inside; leaves are solid. Click a bubble to open it. No layout engine —
// nested flexbox + sqrt sizing keeps it dependency-free and readable.

const MIN = 64;   // px, smallest leaf bubble
const MAX = 220;  // px, largest top-level bubble

function bubbleSize(size: number, biggest: number): number {
  const t = Math.sqrt(size + 1) / Math.sqrt(biggest + 1);   // sqrt so area ~ content, not radius
  return Math.round(MIN + t * (MAX - MIN));
}

function Bubble({ node, biggest, onOpen }: {
  readonly node: MemoryTreeNode;
  readonly biggest: number;
  readonly onOpen: (key: string) => void;
}): JSX.Element {
  const chunked = node.kind === "chunked" && (node.children?.length ?? 0) > 0;
  const px = bubbleSize(node.size, biggest);
  return (
    <div className={`bubble ${chunked ? "chunked" : "leaf"}`}
      style={chunked ? undefined : { width: px, height: px }}
      title={`${node.key} · ${node.label}\n${node.summary}`}
      onClick={(e) => { e.stopPropagation(); onOpen(node.key); }}>
      <div className="bubble-label">{node.label}</div>
      {!chunked && <div className="bubble-size">{node.size} msg</div>}
      {chunked && (
        <div className="bubble-kids">
          {node.children?.map((c) => <Bubble key={c.key} node={c} biggest={biggest} onOpen={onOpen} />)}
        </div>
      )}
    </div>
  );
}

export default function MemoryMap({ tree, onOpen }: {
  readonly tree: readonly MemoryTreeNode[];
  readonly onOpen: (key: string) => void;
}): JSX.Element {
  // scale bubbles against the biggest thing at any level so nesting stays legible
  let biggest = 1;
  const walk = (nodes: readonly MemoryTreeNode[]): void => {
    for (const n of nodes) { biggest = Math.max(biggest, n.size); if (n.children) walk(n.children); }
  };
  walk(tree);
  if (tree.length === 0) return <div className="hint">no crystallized memories yet.</div>;
  return (
    <div className="bubble-map">
      {tree.map((n) => <Bubble key={n.key} node={n} biggest={biggest} onOpen={onOpen} />)}
    </div>
  );
}
