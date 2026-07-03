from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Literal

PermissionMode = Literal["allow", "ask", "deny"]


@dataclass
class Decision:
    allowed: bool
    reason: str


@dataclass
class PermissionPolicy:
    default: PermissionMode = "ask"
    modes: dict[str, PermissionMode] = field(default_factory=dict)
    asker: Callable[[str, dict], bool] | None = None

    def mode_for(self, tool_name: str) -> PermissionMode:
        return self.modes.get(tool_name, self.default)

    def check(self, tool_name: str, args: dict) -> Decision:
        mode = self.modes.get(tool_name, self.default)
        if mode == "allow":
            return Decision(True, "allowed by policy")
        if mode == "deny":
            return Decision(False, "denied by policy")
        if self.asker is None:
            return Decision(False, "ask mode with no approver -> denied")
        return (Decision(True, "approved") if self.asker(tool_name, args)
                else Decision(False, "rejected by user"))
