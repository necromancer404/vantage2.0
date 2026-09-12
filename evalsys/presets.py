from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from evalsys.config import ROOT, AppConfig, LLMSettings, llm_from_dict
from evalsys.schemas import AGENT_NAMES


def load_models_file(path: Path | None = None) -> dict[str, Any]:
    models_path = path or (ROOT / "config" / "models.yaml")
    raw = yaml.safe_load(models_path.read_text(encoding="utf-8")) or {}
    raw.setdefault("presets", {})
    raw.setdefault("test_suite", [])
    return raw


def list_presets(path: Path | None = None) -> dict[str, dict[str, Any]]:
    return dict(load_models_file(path)["presets"])


def suite_names(path: Path | None = None) -> list[str]:
    data = load_models_file(path)
    names = list(data.get("test_suite") or [])
    return names or list(data["presets"].keys())


def preset_settings(name: str, path: Path | None = None) -> LLMSettings:
    presets = list_presets(path)
    if name not in presets:
        known = ", ".join(sorted(presets))
        raise KeyError(f"Unknown preset '{name}'. Known: {known}")
    return llm_from_dict(presets[name])


def apply_preset(config: AppConfig, name: str, path: Path | None = None) -> AppConfig:
    settings = preset_settings(name, path)
    agents = {agent: deepcopy(settings) for agent in AGENT_NAMES}
    return replace(config, llm_default=deepcopy(settings), llm_agents=agents, preset=name)
