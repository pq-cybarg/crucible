from pathlib import Path

from crucible.guardrails.base import GuardrailConfig


class GuardrailStore:
    def __init__(self, path: Path):
        self.path = Path(path)

    def load(self) -> GuardrailConfig:
        if not self.path.exists():
            return GuardrailConfig()
        return GuardrailConfig.model_validate_json(self.path.read_text())

    def save(self, config: GuardrailConfig) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(config.model_dump_json(indent=2))
