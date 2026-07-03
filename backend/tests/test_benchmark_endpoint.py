from fastapi.testclient import TestClient

from crucible.app import create_app
from crucible.registry import Registry


class TimedAdapter:
    num_layers = 2
    def generate_chat(self, messages, n=256):
        return "ok"
    def timed_generate(self, prompt, max_new_tokens=64):
        return {"text": "the ocean is vast and blue", "prompt_tokens": 10,
                "gen_tokens": 40, "prefill_s": 0.1, "decode_s": 2.0}


def test_benchmark_uses_adapter_exact_counts(tmp_path):
    c = TestClient(create_app(registry=Registry(tmp_path / "r.json"),
                              abliteration_adapter=TimedAdapter()))
    r = c.post("/api/runtime/benchmark", json={"tokens": 40}).json()
    assert r["decode_tok_per_s"] == 20.0          # 40 / 2.0
    assert r["prefill_tok_per_s"] == 100.0        # 10 / 0.1
    assert r["gen_tokens"] == 40
    assert "estimated" not in r                    # exact, not estimated


def test_benchmark_needs_a_model(tmp_path):
    c = TestClient(create_app(registry=Registry(tmp_path / "r.json")))
    assert c.post("/api/runtime/benchmark", json={}).status_code == 503
