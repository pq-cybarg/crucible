from crucible.serving import build_command, llama_server_command, vllm_command, BACKENDS


def test_llama_command_has_jinja_for_tools():
    cmd = llama_server_command("/m/x.gguf", 8091)
    assert cmd[0] == "llama-server" and "--jinja" in cmd
    assert "--model" in cmd and "/m/x.gguf" in cmd
    assert cmd[cmd.index("--port") + 1] == "8091"


def test_vllm_command_tensor_parallel_and_tools():
    cmd = vllm_command("/m/hf", 8092, tensor_parallel=4)
    assert cmd[:2] == ["vllm", "serve"]
    assert cmd[cmd.index("--tensor-parallel-size") + 1] == "4"
    assert "--enable-auto-tool-choice" in cmd


def test_vllm_tensor_parallel_min_one():
    cmd = vllm_command("/m/hf", 8092, tensor_parallel=0)
    assert cmd[cmd.index("--tensor-parallel-size") + 1] == "1"


def test_build_command_dispatch():
    assert build_command("llama", "/m/x.gguf", 1)[0] == "llama-server"
    assert build_command("vllm", "/m/hf", 1)[:2] == ["vllm", "serve"]


def test_build_command_unknown_backend():
    try:
        build_command("nope", "/m", 1)
        assert False
    except ValueError:
        pass


def test_backends_registry():
    assert "llama" in BACKENDS and "vllm" in BACKENDS
