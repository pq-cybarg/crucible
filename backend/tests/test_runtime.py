from crucible.runtime import ModelRuntime


class FakeProc:
    def __init__(self): self.terminated = False
    def terminate(self): self.terminated = True


def mkrt(max_resident=1):
    procs = {}
    clock = {"t": 0.0}

    def launcher(mid, path, port):
        p = FakeProc(); procs[mid] = p
        return p

    def tick():
        clock["t"] += 1.0
        return clock["t"]

    rt = ModelRuntime(launcher=launcher, max_resident=max_resident, clock=tick)
    return rt, procs


def test_ensure_starts_and_is_resident():
    rt, procs = mkrt()
    inst = rt.ensure("a", "/m/a.gguf")
    assert rt.is_resident("a")
    assert inst.endpoint.endswith(str(inst.port))
    assert "a" in procs


def test_round_robin_evicts_lru_when_capped_at_one():
    rt, procs = mkrt(max_resident=1)
    rt.ensure("a", "/m/a.gguf")
    rt.ensure("b", "/m/b.gguf")          # must evict a
    assert rt.is_resident("b") and not rt.is_resident("a")
    assert procs["a"].terminated is True  # a's server was stopped


def test_lru_picks_least_recently_used():
    rt, procs = mkrt(max_resident=2)
    rt.ensure("a", "/m/a.gguf")
    rt.ensure("b", "/m/b.gguf")
    rt.touch("a")                         # a now more recent than b
    rt.ensure("c", "/m/c.gguf")           # evict b (LRU)
    assert rt.is_resident("a") and rt.is_resident("c")
    assert not rt.is_resident("b")
    assert procs["b"].terminated is True


def test_ports_are_unique_then_reused_after_stop():
    rt, _ = mkrt(max_resident=3)
    pa = rt.ensure("a", "/m/a.gguf").port
    pb = rt.ensure("b", "/m/b.gguf").port
    assert pa != pb
    rt.stop("a")
    pc = rt.ensure("c", "/m/c.gguf").port
    assert pc == pa                        # freed port reused


def test_ensure_existing_touches_not_relaunch():
    rt, procs = mkrt(max_resident=2)
    first = rt.ensure("a", "/m/a.gguf")
    proc1 = procs["a"]
    again = rt.ensure("a", "/m/a.gguf")
    assert again is first and procs["a"] is proc1   # same instance, not relaunched
    assert again.last_used > again.started_at


def test_status_reports_active_and_resident():
    rt, _ = mkrt(max_resident=2)
    rt.ensure("a", "/m/a.gguf")
    rt.set_active(["a", "b"])
    st = rt.status()
    assert st["max_resident"] == 2
    assert st["active"] == ["a", "b"]
    ids = [r["model_id"] for r in st["resident"]]
    assert ids == ["a"]
    assert st["resident"][0]["active"] is True


def test_stop_all_terminates_everything():
    rt, procs = mkrt(max_resident=5)
    for m in ("a", "b", "c"):
        rt.ensure(m, f"/m/{m}.gguf")
    rt.stop_all()
    assert all(p.terminated for p in procs.values())
    assert rt.status()["resident"] == []
