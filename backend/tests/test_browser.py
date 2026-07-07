"""Browser-driving tool (Brave via Playwright). Skips where the browser/lib isn't available so CI
stays green; runs the real click-through flow when they are."""
import http.server
import socketserver
import threading

import pytest


def _brave_available() -> bool:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return False
    import os
    from crucible.tools.browser import _brave_path
    return os.path.exists(_brave_path())


pytestmark = pytest.mark.skipif(not _brave_available(), reason="Brave/Playwright not available")


@pytest.fixture
def served(tmp_path):
    (tmp_path / "index.html").write_text(
        "<html><body><h1 id='c'>0</h1>"
        "<button id='inc' onclick=\"c.innerText=+c.innerText+1\">inc</button></body></html>")

    class H(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=str(tmp_path), **k)
        def log_message(self, *a):
            pass

    srv = socketserver.TCPServer(("127.0.0.1", 0), H)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{port}/", tmp_path
    srv.shutdown()


def test_browser_drives_a_live_app(served):
    from crucible.tools.browser import Browser, stop_browser
    url, root = served
    b = Browser(root=str(root))
    try:
        assert b.run(action="goto", url=url).ok
        assert b.run(action="text", selector="#c").output.strip() == "0"
        assert b.run(action="click", selector="#inc").ok       # real DOM click → JS runs
        b.run(action="click", selector="#inc")
        assert b.run(action="text", selector="#c").output.strip() == "2"   # counter really incremented
        shot = b.run(action="screenshot", path="shot.png")
        assert shot.ok and (root / "shot.png").exists()
    finally:
        stop_browser()


def test_browser_arg_guards():
    from crucible.tools.browser import Browser
    b = Browser(root=".")
    assert b.run(action="bogus").ok is False
    assert b.run(action="goto").ok is False                    # missing url
    assert b.run(action="click").ok is False                   # missing selector
