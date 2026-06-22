from crucible.rate_limit import RateLimiter


def test_disabled_when_zero():
    rl = RateLimiter(0)
    assert all(rl.allow("ip") for _ in range(100))


def test_blocks_over_limit():
    t = [0.0]
    rl = RateLimiter(2, window_s=60, clock=lambda: t[0])
    assert rl.allow("a") and rl.allow("a")   # 2 allowed
    assert not rl.allow("a")                  # 3rd blocked


def test_window_resets():
    t = [0.0]
    rl = RateLimiter(1, window_s=10, clock=lambda: t[0])
    assert rl.allow("a")
    assert not rl.allow("a")
    t[0] = 11.0                               # window passed
    assert rl.allow("a")


def test_per_key_isolation():
    rl = RateLimiter(1)
    assert rl.allow("a") and rl.allow("b")    # different keys independent
    assert not rl.allow("a")


def test_endpoint_rate_limited(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from crucible.app import create_app
    from crucible.registry import Registry
    monkeypatch.setenv("CRUCIBLE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CRUCIBLE_RATE_LIMIT", "2")
    c = TestClient(create_app(registry=Registry(tmp_path / "r.json")))
    codes = [c.post("/api/agent/run", json={"messages": []}).status_code for _ in range(4)]
    assert codes.count(429) >= 1  # at least one request blocked after the limit
