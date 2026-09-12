from evalsys.config import load_config
from evalsys.llm.providers import extract_chat_text
from evalsys.presets import apply_mix, apply_preset, list_presets, load_runtime_config, suite_names


def test_named_presets_exist():
    names = set(list_presets())
    for required in {
        "gemini",
        "groq",
        "nvidia-deepseek",
        "nvidia-laguna-xs",
        "nvidia-gemma-4",
        "moonshot-kimi-k3",
        "cerebras",
    }:
        assert required in names
    assert "gemini" in suite_names()


def test_apply_preset_sets_all_agents():
    config = apply_preset(load_config(), "nvidia-gemma-4")
    assert config.preset == "nvidia-gemma-4"
    assert config.llm_default.model == "google/gemma-4-31b-it"
    assert config.llm_default.api_key_env == "NVIDIA_API_KEY"
    for settings in config.llm_agents.values():
        assert settings.model == "google/gemma-4-31b-it"
        assert settings.provider == "nvidia"


def test_apply_mix_uses_different_models():
    config = apply_mix(load_config(), "mixed")
    assert config.preset == "mix:mixed"
    assert config.llm_agents["correctness"].model == "google/gemma-4-31b-it"
    assert config.llm_agents["style"].model == "gemini-2.5-flash"
    assert config.llm_agents["complexity"].model == "gemini-2.5-flash"
    assert config.llm_agents["edge_cases"].model == "qwen-3.8-27b"
    assert config.llm_agents["counteragent"].model == "qwen-3.8-27b"
    assert config.llm_agents["correctness"].provider == "nvidia"
    assert config.llm_agents["style"].provider == "gemini"
    assert config.llm_agents["counteragent"].provider == "cerebras"


def test_runtime_default_mix_and_agent_override():
    config = load_runtime_config(mock=True)
    assert config.preset == "mix:mixed"
    assert config.llm_agents["correctness"].provider == "mock"
    config = load_runtime_config(agent_assignments={"style": "groq"}, mock=True)
    assert config.llm_agents["style"].model == "openai/gpt-oss-120b"


def test_kimi_skips_temperature():
    config = apply_preset(load_config(), "moonshot-kimi-k3")
    assert config.llm_default.send_temperature is False
    assert config.llm_default.extra_body["reasoning_effort"] == "low"


def test_extract_chat_text_from_parts():
    payload = {
        "choices": [
            {
                "message": {
                    "content": [{"type": "text", "text": '{"score": 1}'}],
                }
            }
        ]
    }
    assert extract_chat_text(payload) == '{"score": 1}'
