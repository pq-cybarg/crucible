// Render the repo markdown into a static wiki under frontend/dist/docs/ (foundry theme).
import { readFileSync, writeFileSync, mkdirSync, existsSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { marked } from "marked";

const root = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
const out = join(root, "frontend", "dist", "docs");
mkdirSync(out, { recursive: true });

const pages = [
  { md: "README.md", title: "Overview", slug: "index" },
  { md: "PLATFORMS.md", title: "Platforms & Builds", slug: "platforms" },
  { md: "BACKLOG.md", title: "Backlog", slug: "backlog" },
  { md: "docs/superpowers/specs/2026-06-19-crucible-design.md", title: "Design Spec", slug: "design" },
];

const CSS = `
:root{--void:#08090b;--graphite:#0e1013;--steel:#14171c;--line:rgba(255,255,255,.08);--ash:#767d87;--text:#aeb6c1;--bone:#e9edf2;--amber:#ff6a1a;--amber2:#ff8c3f}
*{box-sizing:border-box}body{margin:0;background:var(--void);color:var(--text);font:15px/1.65 "IBM Plex Mono",ui-monospace,monospace}
.wrap{max-width:1000px;margin:0 auto;padding:0 20px}
.docnav{display:flex;gap:18px;align-items:center;flex-wrap:wrap;padding:16px 0;border-bottom:1px solid var(--line);position:sticky;top:0;background:var(--void);z-index:10}
.docnav a{color:var(--ash);text-decoration:none;font-size:13px;letter-spacing:.04em}
.docnav a:hover{color:var(--amber2)}
.docnav .brand{color:var(--amber);font-weight:700;letter-spacing:.12em;font-family:"Chakra Petch",sans-serif}
.doc{padding:28px 0 80px}
.doc h1,.doc h2,.doc h3{font-family:"Chakra Petch",sans-serif;color:var(--bone);letter-spacing:.02em}
.doc h1{font-size:30px;border-bottom:1px solid var(--line);padding-bottom:10px}
.doc h2{font-size:21px;margin-top:34px;color:var(--amber2)}
.doc a{color:var(--amber2)} .doc code{background:var(--steel);padding:1px 6px;border-radius:3px;color:var(--amber2);font-size:13px}
.doc pre{background:var(--steel);border:1px solid var(--line);border-radius:5px;padding:14px;overflow:auto}
.doc pre code{background:none;color:var(--text)}
.doc table{border-collapse:collapse;width:100%;margin:14px 0} .doc th,.doc td{border:1px solid var(--line);padding:8px 11px;text-align:left;font-size:13px}
.doc th{color:var(--ash);text-transform:uppercase;font-size:11px;letter-spacing:.1em} .doc blockquote{border-left:2px solid var(--amber);margin:0;padding-left:14px;color:var(--ash)}
`;
const nav = (base) => pages.map((p) => `<a href="${base}${p.slug}.html">${p.title}</a>`).join("");
const tmpl = (title, body) => `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com"><link href="https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@600;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<title>Crucible · ${title}</title><style>${CSS}</style></head><body><div class="wrap">
<nav class="docnav"><a class="brand" href="../">◢ CRUCIBLE</a><a href="../?demo">▶ Live demo</a>${nav("")}<a href="https://github.com/pq-cybarg/crucible">GitHub ↗</a></nav>
<main class="doc">${body}</main></div></body></html>`;

let made = 0;
for (const p of pages) {
  const path = join(root, p.md);
  if (!existsSync(path)) continue;
  writeFileSync(join(out, `${p.slug}.html`), tmpl(p.title, marked.parse(readFileSync(path, "utf8"))));
  made++;
}
console.log(`docs: rendered ${made} pages -> ${out}`);
