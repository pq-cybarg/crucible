import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    models_dir: Path
    host: str = "127.0.0.1"
    port: int = 8400

    @property
    def registry_path(self) -> Path:
        return self.data_dir / "registry.json"


def get_settings() -> Settings:
    data_dir = Path(os.environ.get("CRUCIBLE_DATA_DIR", Path.home() / ".crucible"))
    models_dir = Path(os.environ.get("CRUCIBLE_MODELS_DIR", "models"))
    data_dir.mkdir(parents=True, exist_ok=True)
    return Settings(data_dir=data_dir, models_dir=models_dir)
