"""统一从环境变量读取路径，保证同一份代码在本地和远端都能跑。"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name) or default).expanduser()


DATA_ROOT = _env_path("DATA_ROOT", REPO_ROOT / "data")
MODEL_ROOT = _env_path("MODEL_ROOT", REPO_ROOT / "models")
OUTPUT_ROOT = _env_path("OUTPUT_ROOT", REPO_ROOT / "outputs")
