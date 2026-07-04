from fastapi.testclient import TestClient
from crucible.app import create_app
from crucible.registry import Registry


def mkapp(tmp_path):
    return TestClient(create_app(registry=Registry(tmp_path / "r.json"), agent_root=tmp_path))


def test_tools_catalog_lists_schemas(tmp_path):
    tools = mkapp(tmp_path).get("/api/tools").json()["tools"]
    names = {t["function"]["name"] for t in tools}
    assert {"read_file", "bash", "web_search", "list_dir"} <= names


def test_tools_invoke_runs(tmp_path):
    (tmp_path / "a.txt").write_text("hi")
    (tmp_path / "sub").mkdir()
    r = mkapp(tmp_path).post("/api/tools/invoke", json={"name": "list_dir", "args": {"path": "."}})
    assert r.status_code == 200 and r.json()["ok"] is True
    assert "a.txt" in r.json()["output"]


def test_tools_invoke_unknown_404(tmp_path):
    assert mkapp(tmp_path).post("/api/tools/invoke", json={"name": "ghost"}).status_code == 404


def test_tools_invoke_deny_403(tmp_path):
    r = mkapp(tmp_path).post("/api/tools/invoke",
                             json={"name": "bash", "args": {"command": "echo x"}, "permission": "deny"})
    assert r.status_code == 403
