from evalsys.config import load_config
from evalsys.llm.providers import extract_chat_text
from evalsys.presets import apply_preset, list_presets, suite_names


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
