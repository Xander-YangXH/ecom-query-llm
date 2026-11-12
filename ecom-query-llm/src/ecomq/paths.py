"""统一从环境变量读取路径，保证同一份代码在本地和远端都能跑。"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name) or default).expanduser()


DATA_ROOT = _env_path("DATA_ROOT", REPO_ROOT / "data")
MODEL_ROOT = _env_path("MODEL_ROOT", REPO_ROOT / "models")
OUTPUT_ROOT = _env_path("OUTPUT_ROOT", REPO_ROOT / "outputs")


def redact(path) -> str:
    """把路径中与机器相关的前缀替换成变量名，避免把用户名、主机目录写进报告（DEVLOG P14）。"""
    s = str(path)
    for name, root in (("$DATA_ROOT", DATA_ROOT), ("$OUTPUT_ROOT", OUTPUT_ROOT), ("$MODEL_ROOT", MODEL_ROOT)):
        s = s.replace(str(root), name)
    s = s.replace(str(Path.home()), "~")
    if s.startswith(("/private/", "/tmp/", "/var/", "/Users/", "/home/")):
        s = Path(s).name  # 其他机器相关的绝对路径只保留文件名
    return s
