"""Small bridge between the new attack package and the unmodified AEGIS tree."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AEGIS_DIR = ROOT / "aegis"


def ensure_aegis_path() -> None:
    """Make the flat AEGIS modules importable without editing AEGIS itself."""
    if str(AEGIS_DIR) not in sys.path:
        sys.path.insert(0, str(AEGIS_DIR))


def configure_offline(hf_home: str | None = None) -> None:
    """Prefer a local Hugging Face cache; avoid accidental network access."""
    if hf_home:
        os.environ.setdefault("HF_HOME", hf_home)
        os.environ.setdefault("HF_HUB_CACHE", str(Path(hf_home) / "hub"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    # Multilingual audio needs the extended refusal token set; the AEGIS default
    # is English-only.
    os.environ.setdefault("REFUSAL_PHRASE_SET", "multi")


def get_adapter(name: str = "qwen2_audio"):
    ensure_aegis_path()
    import model_registry as registry

    return registry.get(name)


def registry_module():
    ensure_aegis_path()
    import model_registry as registry

    return registry


def aegis_common():
    ensure_aegis_path()
    import common

    return common
