from crucible.tools.search import Glob, Grep


def test_glob_finds_files(tmp_path):
    (tmp_path / "a.py").write_text("x"); (tmp_path / "b.txt").write_text("y")
    out = Glob(root=tmp_path).run(pattern="*.py").output
    assert "a.py" in out and "b.txt" not in out


def test_grep_finds_lines(tmp_path):
    (tmp_path / "a.py").write_text("alpha\nbeta\nalpha2\n")
    out = Grep(root=tmp_path).run(pattern="alpha").output
    assert "a.py:1:alpha" in out and "a.py:3:alpha2" in out and "beta" not in out
