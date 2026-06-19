import stat
import sys
import textwrap

import pytest


@pytest.fixture
def fake_llama_server(tmp_path):
    """A stand-in 'llama-server' that serves /health then sleeps, so start/stop is real but cheap."""
    script = tmp_path / "fake-llama-server"
    script.write_text(textwrap.dedent(f"""\
        #!{sys.executable}
        import sys, http.server, threading, time
        port = int(sys.argv[sys.argv.index("--port") + 1])
        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200); self.end_headers(); self.wfile.write(b'{{"status":"ok"}}')
            def log_message(self, *a): pass
        srv = http.server.HTTPServer(("127.0.0.1", port), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        while True: time.sleep(0.2)
    """))
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return str(script)
