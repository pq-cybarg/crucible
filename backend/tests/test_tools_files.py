from crucible.tools.files import EditFile, ReadFile, WriteFile


def test_write_then_read(tmp_path):
    w = WriteFile(root=tmp_path); r = ReadFile(root=tmp_path)
    assert w.run(path="a.txt", content="hello").ok
    assert r.run(path="a.txt").output == "hello"


def test_read_missing_errors(tmp_path):
    res = ReadFile(root=tmp_path).run(path="nope.txt")
    assert res.ok is False and res.error


def test_edit_unique_replace(tmp_path):
    WriteFile(root=tmp_path).run(path="a.txt", content="foo bar foo")
    res = EditFile(root=tmp_path).run(path="a.txt", old="bar", new="baz")
    assert res.ok
    assert ReadFile(root=tmp_path).run(path="a.txt").output == "foo baz foo"


def test_edit_nonunique_errors(tmp_path):
    WriteFile(root=tmp_path).run(path="a.txt", content="x x")
    res = EditFile(root=tmp_path).run(path="a.txt", old="x", new="y")
    assert res.ok is False


def test_escape_root_blocked(tmp_path):
    res = ReadFile(root=tmp_path).run(path="../../etc/passwd")
    assert res.ok is False and "outside" in (res.error or "")
