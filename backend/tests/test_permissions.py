from crucible.permissions import PermissionPolicy


def test_allow_mode():
    assert PermissionPolicy(modes={"read_file": "allow"}).check("read_file", {}).allowed


def test_deny_mode():
    assert PermissionPolicy(modes={"bash": "deny"}).check("bash", {}).allowed is False


def test_ask_uses_callback():
    pol = PermissionPolicy(default="ask", asker=lambda name, args: name == "read_file")
    assert pol.check("read_file", {}).allowed is True
    assert pol.check("bash", {}).allowed is False


def test_ask_without_asker_denies():
    assert PermissionPolicy(default="ask").check("bash", {}).allowed is False
