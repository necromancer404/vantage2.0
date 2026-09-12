from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from evalsys.config import ROOT, AppConfig, LLMSettings, llm_from_dict, load_config
from evalsys.schemas import AGENT_NAMES


def load_models_file(path: Path | None = None) -> dict[str, Any]:
    models_path = path or (ROOT / "config" / "models.yaml")
    raw = yaml.safe_load(models_path.read_text(encoding="utf-8")) or {}
    raw.setdefault("presets", {})
    raw.setdefault("mixes", {})
    raw.setdefault("test_suite", [])
    return raw


def list_presets(path: Path | None = None) -> dict[str, dict[str, Any]]:
    return dict(load_models_file(path)["presets"])


def list_mixes(path: Path | None = None) -> dict[str, dict[str, Any]]:
    return dict(load_models_file(path)["mixes"])


def suite_names(path: Path | None = None) -> list[str]:
    data = load_models_file(path)
    names = list(data.get("test_suite") or [])
    return names or list(data["presets"].keys())


def yaml_mix_name(llm_path: Path | None = None) -> str:
    raw = yaml.safe_load((llm_path or ROOT / "config" / "llm.yaml").read_text(encoding="utf-8")) or {}
    return str(raw.get("mix") or "").strip()


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


def mix_assignments(name: str, path: Path | None = None) -> dict[str, str]:
    mixes = list_mixes(path)
    if name not in mixes:
        known = ", ".join(sorted(mixes))
        raise KeyError(f"Unknown mix '{name}'. Known: {known}")
    agents = dict((mixes[name].get("agents") or {}))
    missing = [agent for agent in AGENT_NAMES if not agents.get(agent)]
    if missing:
        raise KeyError(f"Mix '{name}' is missing agents: {', '.join(missing)}")
    return {agent: str(agents[agent]) for agent in AGENT_NAMES}


def assign_agents(config: AppConfig, assignments: dict[str, str], path: Path | None = None) -> AppConfig:
    agents = dict(config.llm_agents)
    for agent, preset_name in assignments.items():
        if agent not in AGENT_NAMES:
            raise KeyError(f"Unknown agent '{agent}'. Known: {', '.join(AGENT_NAMES)}")
        agents[agent] = deepcopy(preset_settings(preset_name, path))
    label = config.preset
    parts = [f"{agent}={assignments[agent]}" for agent in AGENT_NAMES if agent in assignments]
    if parts:
        label = "mix:" + ",".join(parts) if not label.startswith("mix:") else label
    return replace(config, llm_agents=agents, preset=label)


def apply_mix(config: AppConfig, name: str, path: Path | None = None) -> AppConfig:
    assignments = mix_assignments(name, path)
    agents = {agent: deepcopy(preset_settings(preset, path)) for agent, preset in assignments.items()}
    return replace(config, llm_default=deepcopy(agents["correctness"]), llm_agents=agents, preset=f"mix:{name}")


def load_runtime_config(
    *,
    mix: str = "",
    preset: str = "",
    agent_assignments: dict[str, str] | None = None,
    mock: bool = False,
) -> AppConfig:
    config = load_config()
    chosen_mix = mix or ("" if preset else yaml_mix_name())
    if preset:
        config = apply_preset(config, preset)
    elif chosen_mix:
        config = apply_mix(config, chosen_mix)
    if agent_assignments:
        config = assign_agents(config, agent_assignments)
        if chosen_mix and not preset:
            config = replace(config, preset=f"mix:{chosen_mix}+overrides")
    if mock:
        config.llm_default.provider = "mock"
        config.preset = config.preset or "mock"
        for settings in config.llm_agents.values():
            settings.provider = "mock"
    return config


def describe_agents(config: AppConfig) -> list[str]:
    lines = []
    for name in AGENT_NAMES:
        settings = config.llm_agents[name]
        lines.append(f"{name}: {settings.provider} / {settings.model}")
    return lines
