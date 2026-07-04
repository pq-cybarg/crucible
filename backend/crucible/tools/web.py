from __future__ import annotations
import html as _html
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


_RESULT = re.compile(r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.DOTALL | re.IGNORECASE)
_SNIPPET = re.compile(r'class="result__snippet"[^>]*>(.*?)</a>', re.DOTALL | re.IGNORECASE)


def parse_ddg_results(html: str, limit: int = 8) -> list[dict]:
    """Parse DuckDuckGo HTML results into {title, url, snippet}. Pure."""
    links = _RESULT.findall(html)
    snippets = _SNIPPET.findall(html)
    out = []
    for i, (url, title) in enumerate(links[:limit]):
        snip = snippets[i] if i < len(snippets) else ""
        clean = lambda x: _html.unescape(_TAG.sub("", x)).strip()
        out.append({"url": clean(url), "title": clean(title), "snippet": clean(snip)})
    return out


class WebSearch:
    name = "web_search"
    description = "Search the web for a query and return the top result titles, URLs, and snippets."
    parameters = {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}

    def __init__(self, root=None, timeout: float = 20.0):
        self.timeout = timeout

    def run(self, query: str) -> ToolResult:
        if not str(query).strip():
            return ToolResult(ok=False, output="", error="empty query")
        try:
            import httpx
            r = httpx.get("https://html.duckduckgo.com/html/", params={"q": query},
                          timeout=self.timeout, follow_redirects=True,
                          headers={"User-Agent": "Mozilla/5.0 Crucible/1.0"})
            r.raise_for_status()
            results = parse_ddg_results(r.text)
            if not results:
                return ToolResult(ok=True, output="(no results)")
            lines = [f"{i+1}. {x['title']}\n   {x['url']}\n   {x['snippet']}" for i, x in enumerate(results)]
            return ToolResult(ok=True, output="\n".join(lines))
        except Exception as e:
            return ToolResult(ok=False, output="", error=f"search failed: {e}")
