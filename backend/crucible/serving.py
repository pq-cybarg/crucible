from __future__ import annotations
# Serving backends. The runtime can launch a model on llama.cpp (GGUF, single box, great on
# Apple/Metal and CPU) or vLLM (HF weights, GPU, high throughput + tensor parallelism across
# GPUs). These build the launch command lines; command construction is pure and unit-tested,
# the actual spawn is a subprocess in the runtime.

BACKENDS = {
    "llama": {"desc": "llama.cpp / llama-server — GGUF, CPU/Metal/CUDA, OpenAI-compatible + tools (--jinja)"},
    "vllm": {"desc": "vLLM — HF weights, CUDA, high throughput, tensor-parallel across GPUs"},
}


def llama_server_command(model_path: str, port: int, host: str = "127.0.0.1",
                         ctx_size: int = 8192, n_gpu_layers: int = 999,
                         jinja: bool = True) -> list[str]:
    """llama-server args. --jinja keeps OpenAI tool-calling working for capable chat templates."""
    cmd = ["llama-server", "--model", model_path, "--port", str(port), "--host", host,
           "--ctx-size", str(ctx_size), "--n-gpu-layers", str(n_gpu_layers)]
    if jinja:
        cmd.append("--jinja")
    return cmd


def vllm_command(model_path: str, port: int, host: str = "127.0.0.1",
                 tensor_parallel: int = 1, gpu_memory_utilization: float = 0.9,
                 max_model_len: int | None = None, enable_tools: bool = True) -> list[str]:
    """`vllm serve` args. tensor_parallel shards the model across N GPUs (needs N visible GPUs).
    Exposes an OpenAI-compatible server with tool-calling enabled."""
    cmd = ["vllm", "serve", model_path, "--port", str(port), "--host", host,
           "--tensor-parallel-size", str(max(1, int(tensor_parallel))),
           "--gpu-memory-utilization", str(gpu_memory_utilization)]
    if max_model_len:
        cmd += ["--max-model-len", str(int(max_model_len))]
    if enable_tools:
        cmd += ["--enable-auto-tool-choice", "--tool-call-parser", "hermes"]
    return cmd


def build_command(backend: str, model_path: str, port: int, host: str = "127.0.0.1",
                  tensor_parallel: int = 1) -> list[str]:
    """Dispatch to the right backend's command builder."""
    if backend == "vllm":
        return vllm_command(model_path, port, host, tensor_parallel=tensor_parallel)
    if backend == "llama":
        return llama_server_command(model_path, port, host)
    raise ValueError(f"unknown serving backend: {backend}")
