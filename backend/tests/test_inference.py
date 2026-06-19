from crucible.inference import LlamaServer, wait_healthy


def test_command_construction():
    srv = LlamaServer(model_path="/m/x.gguf", port=8081, ctx=8192, gpu_layers=50)
    cmd = srv.command()
    assert cmd[0] == "llama-server"
    assert "--model" in cmd and "/m/x.gguf" in cmd
    assert "--port" in cmd and "8081" in cmd
    assert "--ctx-size" in cmd and "8192" in cmd
    assert "--n-gpu-layers" in cmd and "50" in cmd


def test_endpoint():
    assert LlamaServer("/m/x.gguf", 8081).endpoint == "http://127.0.0.1:8081"


def test_start_stop_and_health(fake_llama_server):
    srv = LlamaServer(model_path="/m/x.gguf", port=8137, binary=fake_llama_server)
    srv.start()
    try:
        assert wait_healthy(srv.endpoint, timeout=10) is True
        assert srv.is_running is True
    finally:
        srv.stop()
    assert srv.is_running is False
