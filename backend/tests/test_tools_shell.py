from crucible.tools.shell import Bash


def test_bash_echo(tmp_path):
    res = Bash(root=tmp_path).run(command="echo hello")
    assert res.ok and "hello" in res.output


def test_bash_nonzero_exit(tmp_path):
    res = Bash(root=tmp_path).run(command="exit 3")
    assert res.ok is False


def test_bash_timeout(tmp_path):
    res = Bash(root=tmp_path, timeout=0.5).run(command="sleep 5")
    assert res.ok is False and res.error == "timeout"
