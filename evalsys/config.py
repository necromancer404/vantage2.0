from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from evalsys.schemas import AGENT_NAMES

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class LLMSettings:
    provider: str
    model: str
    api_key_env: str
    base_url: str | None = None
    base_url_env: str | None = None
    temperature: float = 0.2
    max_tokens: int = 1200
    timeout_seconds: float = 90.0
    extra_headers: dict[str, str] = field(default_factory=dict)
    extra_body: dict[str, Any] = field(default_factory=dict)
    send_temperature: bool = True
    token_field: str = "max_tokens"


@dataclass
class WeightSettings:
    agents: dict[str, float]
    grade_bands: dict[str, float]
    disagreement_spread_penalty: float = 0.15
    max_score: float = 100.0
    min_score: float = 0.0


@dataclass
class AppConfig:
    llm_default: LLMSettings
    llm_agents: dict[str, LLMSettings]
    weights: WeightSettings
    root: Path = ROOT
    preset: str = ""


def deep_merge(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(base)
    if not override:
        return merged
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        elif value is not None:
            merged[key] = value
    return merged


def llm_from_dict(raw: dict[str, Any]) -> LLMSettings:
    token_field = str(raw.get("token_field") or "max_tokens")
    if token_field not in {"max_tokens", "max_completion_tokens"}:
        raise ValueError(f"Unsupported token_field '{token_field}'")
    return LLMSettings(
        provider=str(raw["provider"]),
        model=str(raw["model"]),
        api_key_env=str(raw.get("api_key_env") or ""),
        base_url=raw.get("base_url"),
        base_url_env=raw.get("base_url_env"),
        temperature=float(raw.get("temperature", 0.2)),
        max_tokens=int(raw.get("max_tokens", 1200)),
        timeout_seconds=float(raw.get("timeout_seconds", 90)),
        extra_headers={str(k): str(v) for k, v in dict(raw.get("extra_headers") or {}).items()},
        extra_body=dict(raw.get("extra_body") or {}),
        send_temperature=bool(raw.get("send_temperature", True)),
        token_field=token_field,
    )


def load_config(
    llm_path: Path | None = None,
    weights_path: Path | None = None,
    env_path: Path | None = None,
) -> AppConfig:
    load_dotenv(env_path or (ROOT / ".env"), override=False)

    llm_raw = yaml.safe_load((llm_path or ROOT / "config" / "llm.yaml").read_text(encoding="utf-8"))
    weights_raw = yaml.safe_load((weights_path or ROOT / "config" / "weights.yaml").read_text(encoding="utf-8"))

    default = llm_from_dict(llm_raw["default"])
    agent_overrides = llm_raw.get("agents") or {}
    llm_agents: dict[str, LLMSettings] = {}
    for name in AGENT_NAMES:
        merged = deep_merge(llm_raw["default"], agent_overrides.get(name) or {})
        llm_agents[name] = llm_from_dict(merged)

    agent_weights = {name: float(weights_raw["agents"][name]) for name in AGENT_NAMES}
    total = sum(agent_weights.values())
    if abs(total - 1.0) > 1e-6:
        agent_weights = {k: v / total for k, v in agent_weights.items()}

    weights = WeightSettings(
        agents=agent_weights,
        grade_bands={k: float(v) for k, v in (weights_raw.get("grade_bands") or {}).items()},
        disagreement_spread_penalty=float(weights_raw.get("disagreement_spread_penalty", 0.15)),
        max_score=float(weights_raw.get("max_score", 100)),
        min_score=float(weights_raw.get("min_score", 0)),
    )
    return AppConfig(llm_default=default, llm_agents=llm_agents, weights=weights)
