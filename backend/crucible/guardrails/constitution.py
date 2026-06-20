from typing import Callable

from crucible.guardrails.base import GuardrailAction

Critic = Callable[[str, str], str]


def build_critic_prompt(constitution: str, draft: str) -> str:
    return (
        "Revise the DRAFT so it complies with the CONSTITUTION. "
        "Return only the revised text, nothing else.\n\n"
        f"CONSTITUTION:\n{constitution}\n\nDRAFT:\n{draft}"
    )


class ConstitutionalCritic:
    def __init__(self, constitution: str, critic: Critic):
        self.constitution = constitution
        self.critic = critic

    def revise(self, draft: str) -> tuple[str, GuardrailAction | None]:
        revised = self.critic(self.constitution, draft)
        if revised == draft:
            return draft, None
        return revised, GuardrailAction(layer="constitution", stage="output",
                                        action="revise", detail="revised per constitution")
