from pathlib import Path
from typing import Any

import yaml

# Resolves to turbofan-rul-engine/ regardless of where you run python from
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(config_path: Path | str = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load the YAML configuration file and return it as a dictionary."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at: {path}")

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(relative_path: str) -> Path:
    """Resolve a path from config.yaml relative to the project root."""
    return PROJECT_ROOT / relative_path