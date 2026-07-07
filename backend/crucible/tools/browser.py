from __future__ import annotations
# A real browser-driving tool: navigate, click, type, read, and screenshot a live page — so the agent
# can actually USE the web apps it builds (click through flows, verify behaviour), not just fetch HTML.
# Drives BRAVE (Chromium-based) via Playwright's executable_path — no separate browser download.
#
# Playwright objects are thread-affine and the agent's tool calls can land on different worker threads,
# so the browser lives in ONE dedicated thread with a command queue; every action is dispatched there
# and shares a single persistent page. That's what makes multi-step driving (goto → fill → click →
# read) work across separate tool calls.
import os
import queue
import threading

from crucible.tools.base import ToolResult

_DEFAULT_BRAVE = "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser"


def _brave_path() -> str:
    return os.environ.get("CRUCIBLE_BROWSER_PATH", _DEFAULT_BRAVE)


def _headless() -> bool:
    return os.environ.get("CRUCIBLE_BROWSER_HEADLESS", "1") not in ("0", "false", "no")


class _BrowserWorker:
    """Owns the Playwright browser on its own thread; actions are queued to it and share one page."""

    def __init__(self):
        self._q: "queue.Queue" = queue.Queue()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._error: str | None = None

    def _ensure(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._error = None
            ready = threading.Event()
            self._thread = threading.Thread(target=self._loop, args=(ready,), daemon=True)
            self._thread.start()
            ready.wait(30)
            if self._error:
                raise RuntimeError(self._error)

    def _loop(self, ready: threading.Event) -> None:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            self._error = "playwright not installed (pip install playwright)"
            ready.set()
            return
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(executable_path=_brave_path(), headless=_headless())
                page = browser.new_page()
                ready.set()
                while True:
                    action, args, box = self._q.get()
                    if action == "__stop__":
                        break
                    try:
                        box["result"] = _do(page, action, args)
                    except Exception as e:      # a page error shouldn't kill the browser thread
                        box["error"] = f"{type(e).__name__}: {e}"
                    finally:
                        box["done"].set()
                browser.close()
        except Exception as e:                  # launch failure (e.g. Brave not found)
            self._error = f"browser launch failed: {e}"
            ready.set()

    def call(self, action: str, args: dict, timeout: float = 60.0) -> dict:
        self._ensure()
        box: dict = {"done": threading.Event()}
        self._q.put((action, args, box))
        if not box["done"].wait(timeout):
            return {"error": f"browser action '{action}' timed out"}
        return box


def _snippet(page, limit: int = 1500) -> str:
    try:
        return (page.inner_text("body") or "")[:limit]
    except Exception:
        return ""


def _do(page, action: str, args: dict) -> str:
    """Perform one browser action against the live page; return a short human/agent-readable result."""
    if action == "goto":
        page.goto(args["url"], wait_until="domcontentloaded", timeout=30000)
        return f"navigated to {page.url}\ntitle: {page.title()}\n---\n{_snippet(page)}"
    if action == "click":
        page.click(args["selector"], timeout=10000)
        return f"clicked {args['selector']}\nurl now {page.url}\n---\n{_snippet(page)}"
    if action == "fill":
        page.fill(args["selector"], args.get("text", ""), timeout=10000)
        return f"filled {args['selector']}"
    if action == "text":
        sel = args.get("selector")
        return page.inner_text(sel) if sel else _snippet(page, 8000)
    if action == "content":
        from crucible.tools.web import html_to_text
        return html_to_text(page.content())
    if action == "eval":
        return str(page.evaluate(args["script"]))
    if action == "screenshot":
        path = args.get("path") or "browser-screenshot.png"
        root = args.get("_root")
        full = os.path.join(root, path) if root else path
        os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
        page.screenshot(path=full, full_page=bool(args.get("full_page")))
        return f"saved screenshot to {full} ({os.path.getsize(full)} bytes)"
    if action == "wait":
        page.wait_for_timeout(int(float(args.get("ms", 500))))
        return f"waited {args.get('ms', 500)}ms"
    return f"unknown action '{action}'. Use: goto, click, fill, text, content, eval, screenshot, wait."


_WORKER = _BrowserWorker()

_ACTIONS = ("goto", "click", "fill", "text", "content", "eval", "screenshot", "wait")


class Browser:
    name = "browser"
    description = ("Drive a real browser (Brave) to use a running web app: navigate, click, type, read "
                   "the live page, run JS, or screenshot. One persistent page across calls. Actions: "
                   "goto(url), click(selector), fill(selector,text), text(selector?), content, "
                   "eval(script), screenshot(path?), wait(ms).")
    parameters = {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": list(_ACTIONS)},
            "url": {"type": "string"},
            "selector": {"type": "string"},
            "text": {"type": "string"},
            "script": {"type": "string"},
            "path": {"type": "string"},
            "ms": {"type": "number"},
            "full_page": {"type": "boolean"},
        },
        "required": ["action"],
    }

    def __init__(self, root=None):
        self.root = str(root) if root else "."

    def run(self, action: str = "", **kw) -> ToolResult:
        if action not in _ACTIONS:
            return ToolResult(ok=False, output="", error=f"action must be one of {', '.join(_ACTIONS)}")
        if action == "goto" and not kw.get("url"):
            return ToolResult(ok=False, output="", error="goto requires 'url'")
        if action in ("click", "fill") and not kw.get("selector"):
            return ToolResult(ok=False, output="", error=f"{action} requires 'selector'")
        if action == "eval" and not kw.get("script"):
            return ToolResult(ok=False, output="", error="eval requires 'script'")
        args = {**kw, "_root": self.root}
        box = _WORKER.call(action, args)
        if box.get("error"):
            return ToolResult(ok=False, output="", error=box["error"])
        return ToolResult(ok=True, output=box.get("result", ""))


def stop_browser() -> None:
    """Best-effort shutdown of the browser thread (for tests / server shutdown)."""
    try:
        _WORKER._q.put(("__stop__", {}, {"done": threading.Event()}))
    except Exception:
        pass
