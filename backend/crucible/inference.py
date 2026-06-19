import subprocess
import time

import httpx


class LlamaServer:
    def __init__(self, model_path: str, port: int, ctx: int = 16384,
                 gpu_layers: int = 999, binary: str = "llama-server"):
        self.model_path = model_path
        self.port = port
        self.ctx = ctx
        self.gpu_layers = gpu_layers
        self.binary = binary
        self._proc: subprocess.Popen | None = None

    def command(self) -> list[str]:
        return [
            self.binary,
            "--model", self.model_path,
            "--port", str(self.port),
            "--ctx-size", str(self.ctx),
            "--n-gpu-layers", str(self.gpu_layers),
            "--host", "127.0.0.1",
        ]

    @property
    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> None:
        if self._proc is not None:
            return
        self._proc = subprocess.Popen(self.command())

    def stop(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None

    @property
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None


def wait_healthy(endpoint: str, timeout: float = 120) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if httpx.get(f"{endpoint}/health", timeout=2).status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        time.sleep(0.3)
    return False
