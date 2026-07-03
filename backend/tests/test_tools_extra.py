from pathlib import Path

from crucible.tools import default_registry
from crucible.tools.files import ListDir, MultiEdit
from crucible.tools.plan import TodoWrite
from crucible.tools.web import WebFetch, html_to_text


def test_default_registry_has_full_skillset(tmp_path):
    names = {t.name for t in default_registry(tmp_path).all()}
    assert {"read_file", "write_file", "edit_file", "multi_edit", "list_dir",
            "grep", "glob", "bash", "web_fetch", "todo_write"} <= names


def test_list_dir(tmp_path):
    (tmp_path / "a.txt").write_text("x")
    (tmp_path / "sub").mkdir()
    out = ListDir(tmp_path).run(".")
    assert out.ok
    assert "sub/" in out.output and "a.txt" in out.output
    assert out.output.index("sub/") < out.output.index("a.txt")   # dirs first


def test_list_dir_rejects_escape(tmp_path):
    assert ListDir(tmp_path).run("../..").ok is False


def test_multi_edit_atomic(tmp_path):
    f = tmp_path / "f.txt"
    f.write_text("alpha beta gamma")
    r = MultiEdit(tmp_path).run("f.txt", [{"old": "alpha", "new": "A"}, {"old": "gamma", "new": "G"}])
    assert r.ok and f.read_text() == "A beta G"


def test_multi_edit_rolls_back_on_bad_edit(tmp_path):
    f = tmp_path / "f.txt"
    f.write_text("alpha beta")
    r = MultiEdit(tmp_path).run("f.txt", [{"old": "alpha", "new": "A"}, {"old": "nope", "new": "X"}])
    assert r.ok is False
    assert f.read_text() == "alpha beta"          # nothing written (atomic)


def test_html_to_text_strips_tags_and_scripts():
    html = "<html><head><style>x{}</style><script>bad()</script></head><body><h1>Hi</h1><p>world &amp; peace</p></body></html>"
    txt = html_to_text(html)
    assert "Hi" in txt and "world & peace" in txt
    assert "bad()" not in txt and "<h1>" not in txt


def test_web_fetch_validates_scheme():
    assert WebFetch().run("ftp://x").ok is False
    assert WebFetch().run("notaurl").ok is False


def test_todo_write_tracks_progress():
    t = TodoWrite()
    r = t.run([{"task": "a", "status": "done"}, {"task": "b", "status": "in_progress"}, {"task": "c"}])
    assert r.ok and "1/3 done" in r.output
    assert "[x] a" in r.output and "[~] b" in r.output and "[ ] c" in r.output
    assert len(t.todos) == 3
