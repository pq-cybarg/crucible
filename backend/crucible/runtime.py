from __future__ import annotations
# Model runtime manager. Turns registered local GGUF models on/off, runs several at once
# when memory allows, and — when it's constrained — round-robins: it keeps at most
# `max_resident` models loaded and evicts the least-recently-used one to make room, so the
# "active" models transparently take turns on the RAM. The launcher (spawn a llama-server)
# is injectable, so the load/evict policy is fully unit-tested without spawning anything.
import time
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class Instance:
    model_id: str
    port: int
    endpoint: str
    proc: object = None
    started_at: float = 0.0
    last_used: float = 0.0


# launcher(model_id, model_path, port, backend, tensor_parallel) -> process handle.
Launcher = Callable[..., object]


class ModelRuntime:
    def __init__(self, launcher: Optional[Launcher] = None, max_resident: int = 1,
                 base_port: int = 8090, clock: Callable[[], float] = time.monotonic,
                 host: str = "127.0.0.1"):
        self.max_resident = max(1, int(max_resident))
        self._base_port = base_port
        self._host = host
        self._clock = clock
        self._launcher = launcher or self._default_launcher
        self._resident: dict[str, Instance] = {}
        self._active: set[str] = set()      # models the user marked "active"

    # ---- policy (pure, tested) ------------------------------------------
    def _alloc_port(self) -> int:
        used = {i.port for i in self._resident.values()}
        port = self._base_port
        while port in used:
            port += 1
        return port

    def _evict_candidates(self, need: int) -> list[str]:
        """LRU model_ids to evict so that resident count + need <= max_resident."""
        over = (len(self._resident) + need) - self.max_resident
        if over <= 0:
            return []
        order = sorted(self._resident.values(), key=lambda i: i.last_used)
        return [i.model_id for i in order[:over]]

    # ---- lifecycle ------------------------------------------------------
    def ensure(self, model_id: str, model_path: str, port: Optional[int] = None,
               backend: str = "llama", tensor_parallel: int = 1) -> Instance:
        """Return a running instance for model_id, starting it (and evicting LRU models to
        respect max_resident) if needed. `backend` selects llama.cpp or vLLM. Touches last_used."""
        inst = self._resident.get(model_id)
        if inst is not None:
            inst.last_used = self._clock()
            return inst
        for victim in self._evict_candidates(1):
            self.stop(victim)
        p = port if port is not None else self._alloc_port()
        endpoint = f"http://{self._host}:{p}"
        proc = self._launcher(model_id, model_path, p, backend, tensor_parallel)
        now = self._clock()
        inst = Instance(model_id=model_id, port=p, endpoint=endpoint, proc=proc,
                        started_at=now, last_used=now)
        self._resident[model_id] = inst
        return inst

    def touch(self, model_id: str) -> None:
        inst = self._resident.get(model_id)
        if inst is not None:
            inst.last_used = self._clock()

    def stop(self, model_id: str) -> bool:
        inst = self._resident.pop(model_id, None)
        if inst is None:
            return False
        proc = inst.proc
        term = getattr(proc, "terminate", None)
        if callable(term):
            try:
                term()
            except Exception:
                pass
        return True

    def stop_all(self) -> None:
        for mid in list(self._resident):
            self.stop(mid)

    def set_active(self, ids: list[str]) -> None:
        self._active = set(ids)

    def is_resident(self, model_id: str) -> bool:
        return model_id in self._resident

    def endpoint_for(self, model_id: str) -> Optional[str]:
        inst = self._resident.get(model_id)
        return inst.endpoint if inst else None

    def status(self) -> dict:
        return {
            "max_resident": self.max_resident,
            "resident": [
                {"model_id": i.model_id, "port": i.port, "endpoint": i.endpoint,
                 "active": i.model_id in self._active,
                 "started_at": i.started_at, "last_used": i.last_used}
                for i in sorted(self._resident.values(), key=lambda x: x.last_used, reverse=True)
            ],
            "active": sorted(self._active),
        }

    # ---- real launcher (model path; not unit-tested) --------------------
    def _default_launcher(self, model_id: str, model_path: str, port: int,
                          backend: str = "llama", tensor_parallel: int = 1) -> object:
        import subprocess
        from crucible.serving import build_command
        cmd = build_command(backend, model_path, port, self._host, tensor_parallel)
        return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
