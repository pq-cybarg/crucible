import type { JSX } from "react";

// Every analysis endpoint attaches a `plain` card: a jargon-free, five-field explanation of what
// the technique is, what it found, what it means, and the honest caveat. This renders it so the
// raw numbers always come with a plain-English diagnosis a non-expert can act on. Shared across
// the Analysis and Pipeline panels.
export type Plain = {
  headline?: string;
  what_it_is?: string;
  what_we_found?: string;
  what_it_means?: string;
  caveat?: string;
};

export default function PlainCard({ res }: { readonly res: Record<string, unknown> | null }): JSX.Element | null {
  const p = res?.["plain"] as Plain | undefined;
  if (!p || !p.headline) return null;
  return (
    <div className="plain-card">
      <div className="plain-headline">{p.headline}</div>
      {p.what_it_is && <p className="plain-line"><span className="plain-tag">what it is</span>{p.what_it_is}</p>}
      {p.what_we_found && <p className="plain-line"><span className="plain-tag">what we found</span>{p.what_we_found}</p>}
      {p.what_it_means && <p className="plain-line"><span className="plain-tag">what it means</span>{p.what_it_means}</p>}
      {p.caveat && <p className="plain-line plain-caveat"><span className="plain-tag">caveat</span>{p.caveat}</p>}
    </div>
  );
}
