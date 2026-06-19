import asyncio
from pathlib import Path

from crucible.client import ChatClient
from crucible.config import get_settings
from crucible.inference import LlamaServer, wait_healthy
from crucible.registry import Model, Registry

MODEL_PATH = "models/glm-4-32b/THUDM_GLM-4-32B-0414-Q4_K_M.gguf"


async def main():
    assert Path(MODEL_PATH).exists(), f"missing model: {MODEL_PATH}"
    srv = LlamaServer(model_path=MODEL_PATH, port=8081, ctx=16384, gpu_layers=999)
    srv.start()
    try:
        assert wait_healthy(srv.endpoint, timeout=180), "server never became healthy"
        reg = Registry(get_settings().registry_path)
        if "glm-4-32b" not in [m.id for m in reg.list()]:
            reg.register(Model(id="glm-4-32b", name="GLM-4-32B-0414", base_id=None,
                               path=MODEL_PATH, quant="Q4_K_M", kind="base",
                               endpoint=srv.endpoint, created="2026-06-19", notes="dev model"))
        reply = await ChatClient(srv.endpoint).chat(
            [{"role": "user", "content": "Reply with exactly: Crucible online."}],
            max_tokens=32, temperature=0.0)
        print("MODEL REPLY:", reply)
        assert reply.strip(), "empty reply"
    finally:
        srv.stop()


if __name__ == "__main__":
    asyncio.run(main())
