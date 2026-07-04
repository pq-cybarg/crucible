from crucible.tools.web import WebSearch, parse_ddg_results


SAMPLE = '''
<div><a class="result__a" href="https://example.com/a">First &amp; Best</a>
<a class="result__snippet">A great snippet here</a></div>
<div><a class="result__a" href="https://example.org/b">Second Result</a>
<a class="result__snippet">Another snippet</a></div>
'''


def test_parse_ddg_results():
    rows = parse_ddg_results(SAMPLE)
    assert len(rows) == 2
    assert rows[0]["url"] == "https://example.com/a"
    assert rows[0]["title"] == "First & Best"
    assert "great snippet" in rows[0]["snippet"]


def test_parse_ddg_empty():
    assert parse_ddg_results("<html>nothing</html>") == []


def test_web_search_rejects_empty_query():
    assert WebSearch().run("  ").ok is False


def test_web_search_tool_schema():
    t = WebSearch()
    assert t.name == "web_search" and "query" in t.parameters["properties"]


def test_dispatch_coerces_hallucinated_name(tmp_path):
    # a model asking for 'search_web' should be routed to the real 'web_search'
    from crucible.agent_react import _dispatch_tool
    from crucible.audit import AuditLog
    from crucible.permissions import PermissionPolicy
    from crucible.tools import default_registry
    reg = default_registry(tmp_path)
    events = list(_dispatch_tool(reg, PermissionPolicy(default="allow"),
                                 AuditLog(tmp_path / "a.jsonl"), None, "c1", "search_web",
                                 {"query": "x"}))
    tr = [e for e in events if e.type == "tool_result"][0]
    assert tr.data["name"] == "web_search"        # coerced, not "no such tool"
