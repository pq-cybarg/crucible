from crucible.permissions import PathRule, PermissionPolicy, extract_paths


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


# --- path-scoped rules: limited-permission directories/files ---------------------------------------
def test_path_deny_overrides_allowed_tool():
    # read is broadly allowed, but a rule denies the secrets directory
    pol = PermissionPolicy(modes={"read_file": "allow"},
                           path_rules=[PathRule("~/.ssh/**", "deny")])
    assert pol.check("read_file", {"path": "~/.ssh/id_rsa"}).allowed is False
    assert pol.check("read_file", {"path": "~/projects/notes.md"}).allowed is True


def test_path_allow_inside_workspace_overrides_ask():
    # default ask, but a designated safe workspace auto-allows without prompting
    pol = PermissionPolicy(default="ask", path_rules=[PathRule("/work/**", "allow")])
    assert pol.check("write_file", {"file_path": "/work/out.txt"}).allowed is True
    assert pol.check("write_file", {"file_path": "/etc/passwd"}).allowed is False   # ask, no asker


def test_path_rule_scoped_to_named_tools():
    pol = PermissionPolicy(modes={"read_file": "allow", "bash": "allow"},
                           path_rules=[PathRule("/data/**", "deny", tools=("bash",))])
    assert pol.check("bash", {"command": "cat /data/secret"}).allowed is False   # bash blocked
    assert pol.check("read_file", {"path": "/data/secret"}).allowed is True      # read unaffected


def test_deny_is_decisive_across_multiple_paths():
    # a bash command touching both a safe and a denied path is denied — no half-touch
    pol = PermissionPolicy(modes={"bash": "allow"},
                           path_rules=[PathRule("**/secrets/**", "deny")])
    d = pol.check("bash", {"command": "cp ./secrets/key.pem /tmp/ok"})
    assert d.allowed is False and "denied" in d.reason


def test_extract_paths_from_bash_and_structured_args():
    assert "/etc/hosts" in extract_paths("bash", {"command": "cat /etc/hosts | grep x"})
    assert extract_paths("read_file", {"path": "~/a.txt"}) == ["~/a.txt"]
    assert extract_paths("read_file", {}) == []
