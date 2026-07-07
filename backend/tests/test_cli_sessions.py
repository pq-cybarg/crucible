"""Project-local sessions: sessions live in <project>/.crucible and travel with the folder (unlike a
global store keyed by absolute path), a project can hold many browseable sessions, and --continue /
--resume find the most recent."""
import os

import pytest

from crucible import cli


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.delenv("CRUCIBLE_HOME", raising=False)
    root = tmp_path / "myproj"
    (root / ".git").mkdir(parents=True)          # a project marker
    sub = root / "src" / "deep"
    sub.mkdir(parents=True)
    monkeypatch.chdir(sub)                        # run from deep inside the project
    return root


def test_project_root_found_from_subdir(project):
    assert cli.project_root() == project.resolve()


def test_store_is_inside_the_project(project):
    # the session store is <project>/.crucible — inside the folder, so it moves with it
    assert cli.store_dir() == project.resolve() / ".crucible"
    cli.save_session("alpha", [{"role": "user", "content": "hi"}])
    assert (project / ".crucible" / "sessions" / "alpha.json").exists()


def test_crucible_home_env_forces_global_store(tmp_path, monkeypatch):
    monkeypatch.setenv("CRUCIBLE_HOME", str(tmp_path / "global"))
    monkeypatch.chdir(tmp_path)
    assert cli.store_dir() == tmp_path / "global"


def test_most_recent_and_browse(project, monkeypatch):
    import time
    cli.save_session("first", [{"role": "user", "content": "the first task"}])
    time.sleep(0.01)
    cli.save_session("second", [{"role": "user", "content": "later work"},
                                {"role": "assistant", "content": "ok"}])
    # --continue resumes the newest
    assert cli.most_recent_session() == "second"
    # browsing shows metadata + preview, newest first — a project holds many sessions
    rows = cli.browse_sessions()
    assert [r["name"] for r in rows] == ["second", "first"]
    assert rows[0]["msgs"] == 2 and rows[0]["preview"] == "later work"
    assert rows[1]["preview"] == "the first task"


def test_sessions_are_isolated_per_project(tmp_path, monkeypatch):
    monkeypatch.delenv("CRUCIBLE_HOME", raising=False)
    for name in ("projA", "projB"):
        p = tmp_path / name
        (p / ".git").mkdir(parents=True)
        monkeypatch.chdir(p)
        cli.save_session("s", [{"role": "user", "content": name}])
    # each project only sees its own session — no global bleed
    monkeypatch.chdir(tmp_path / "projA")
    assert cli.load_session("s")[0]["content"] == "projA"
    monkeypatch.chdir(tmp_path / "projB")
    assert cli.load_session("s")[0]["content"] == "projB"
