from __future__ import annotations
import fnmatch
import os
from dataclasses import dataclass, field
from typing import Callable, Literal

PermissionMode = Literal["allow", "ask", "deny"]

# Tool argument keys that name a filesystem path — so path-scoped rules can apply to the RIGHT tools
# (read/write/edit/ls/grep/…) without the caller wiring each one. Bash is freeform, handled below.
_PATH_KEYS = ("path", "file", "file_path", "filepath", "dir", "directory", "cwd", "target")


@dataclass
class Decision:
    allowed: bool
    reason: str


@dataclass
class PathRule:
    """Scope a tool's permission to files/directories. `glob` is fnmatch-style, matched against the
    absolute, ~-expanded path a tool is about to touch (e.g. '~/.ssh/**', '/etc/*', '**/secrets*').
    `tools` limits the rule to named tools (empty = every path-taking tool). Rules are evaluated in
    order, first match wins — like firewall rules — so put specific denies before broad allows."""
    glob: str
    mode: PermissionMode
    tools: tuple[str, ...] = ()

    def _abs(self, p: str) -> str:
        return os.path.abspath(os.path.expanduser(p))

    def matches(self, tool_name: str, path: str) -> bool:
        if self.tools and tool_name not in self.tools:
            return False
        pat = os.path.expanduser(self.glob)
        cand = self._abs(path)
        # match against both the raw and the absolute path so '~/.ssh/**' and '/etc/*' both work,
        # and add a '/**' form so a directory glob also matches files beneath it.
        pats = {pat, self._abs(pat)}
        return any(fnmatch.fnmatch(path, pt) or fnmatch.fnmatch(cand, pt)
                   or fnmatch.fnmatch(cand, pt.rstrip("/") + "/**") for pt in pats)


def extract_paths(tool_name: str, args: dict) -> list[str]:
    """Best-effort: the filesystem paths a tool call would touch. Recognized path arguments for
    structured tools; for bash, the whitespace tokens of the command that look like paths (contain a
    '/' or start with '~' or '.') so a deny on a sensitive directory still bites a shell command."""
    paths: list[str] = []
    for k in _PATH_KEYS:
        v = args.get(k)
        if isinstance(v, str) and v:
            paths.append(v)
    if tool_name in ("bash", "shell"):
        cmd = args.get("command") or args.get("cmd") or ""
        if isinstance(cmd, str):
            for tok in cmd.replace("'", " ").replace('"', " ").split():
                if "/" in tok or tok.startswith("~") or tok.startswith("."):
                    paths.append(tok)
    return paths


@dataclass
class PermissionPolicy:
    default: PermissionMode = "ask"
    modes: dict[str, PermissionMode] = field(default_factory=dict)
    path_rules: list[PathRule] = field(default_factory=list)
    asker: Callable[[str, dict], bool] | None = None

    def mode_for(self, tool_name: str) -> PermissionMode:
        return self.modes.get(tool_name, self.default)

    def _path_mode(self, tool_name: str, args: dict) -> tuple[PermissionMode | None, str]:
        """The first path rule that matches any path this call touches wins (firewall order). A deny
        anywhere is decisive even if a later path would allow — you can't half-touch a sensitive set."""
        if not self.path_rules:
            return None, ""
        paths = extract_paths(tool_name, args)
        # a deny match is decisive regardless of order, so a broad allow can't leak a denied path.
        for path in paths:
            for rule in self.path_rules:
                if rule.mode == "deny" and rule.matches(tool_name, path):
                    return "deny", f"path '{path}' denied by rule '{rule.glob}'"
        for path in paths:
            for rule in self.path_rules:
                if rule.matches(tool_name, path):
                    return rule.mode, f"path '{path}' -> rule '{rule.glob}' ({rule.mode})"
        return None, ""

    def check(self, tool_name: str, args: dict) -> Decision:
        # path rules override the per-tool mode: a tool may be broadly allowed yet denied on ~/.ssh,
        # or broadly ask yet auto-allowed inside a designated safe workspace.
        scoped, why = self._path_mode(tool_name, args)
        mode = scoped if scoped is not None else self.modes.get(tool_name, self.default)
        if mode == "allow":
            return Decision(True, why or "allowed by policy")
        if mode == "deny":
            return Decision(False, why or "denied by policy")
        if self.asker is None:
            return Decision(False, "ask mode with no approver -> denied")
        return (Decision(True, "approved") if self.asker(tool_name, args)
                else Decision(False, "rejected by user"))
