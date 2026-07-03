from __future__ import annotations
import re

from crucible.tools.base import ToolResult

_TAG = re.compile(r"<[^>]+>")
_SCRIPT = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE)
_WS = re.compile(r"\n\s*\n\s*\n+")


def html_to_text(html: str, limit: int = 20000) -> str:
    """Strip scripts/styles/tags to readable text, collapse blank runs, cap length. Pure."""
    s = _SCRIPT.sub(" ", html)
    s = _TAG.sub("", s)
    s = (s.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<")
         .replace("&gt;", ">").replace("&quot;", '"').replace("&#39;", "'"))
    s = _WS.sub("\n\n", s).strip()
    return s[:limit] + ("\n…(truncated)" if len(s) > limit else "")


class WebFetch:
    name = "web_fetch"
    description = "Fetch a URL over HTTP(S) and return its text content (HTML stripped to readable text)."
    parameters = {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}

    def __init__(self, root=None, timeout: float = 20.0):
        self.timeout = timeout

    def run(self, url: str) -> ToolResult:
        if not re.match(r"^https?://", url):
            return ToolResult(ok=False, output="", error="url must start with http:// or https://")
        try:
            import httpx
            r = httpx.get(url, timeout=self.timeout, follow_redirects=True,
                          headers={"User-Agent": "Crucible/1.0"})
            r.raise_for_status()
            ct = r.headers.get("content-type", "")
            text = html_to_text(r.text) if "html" in ct else r.text[:20000]
            return ToolResult(ok=True, output=text)
        except Exception as e:  # network errors, bad status, etc.
            return ToolResult(ok=False, output="", error=f"fetch failed: {e}")
