from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


CONFIG_DIRNAME = ".attention_kernel_challenge"
CONFIG_FILENAME = "backend.json"


@dataclass(frozen=True)
class ModalBackendConfig:
    gpu: str = "H100!:1"
    timeout_s: int = 120
    python_version: str = "3.11"


@dataclass(frozen=True)
class HarnessConfig:
    default_backend: str
    modal: ModalBackendConfig = ModalBackendConfig()


def repo_root_from_file(module_file: str) -> Path:
    return Path(module_file).resolve().parent.parent


def config_path(repo_root: Path) -> Path:
    return repo_root / CONFIG_DIRNAME / CONFIG_FILENAME


def load_config(repo_root: Path) -> Optional[HarnessConfig]:
    path = config_path(repo_root)
    if not path.exists():
        return None

    payload = json.loads(path.read_text())
    modal_payload = payload.get("modal", {})
    return HarnessConfig(
        default_backend=payload["default_backend"],
        modal=ModalBackendConfig(
            gpu=modal_payload.get("gpu", "H100!:1"),
            timeout_s=int(modal_payload.get("timeout_s", 120)),
            python_version=modal_payload.get("python_version", "3.11"),
        ),
    )


def save_config(repo_root: Path, config: HarnessConfig) -> Path:
    path = config_path(repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(config), indent=2) + "\n")
    return path


def clear_config(repo_root: Path) -> None:
    path = config_path(repo_root)
    if path.exists():
        path.unlink()
    if path.parent.exists() and not any(path.parent.iterdir()):
        path.parent.rmdir()
