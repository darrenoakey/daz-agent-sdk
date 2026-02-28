from __future__ import annotations

from pathlib import Path

import yaml

from daz_agent_sdk.types import Tier
from daz_agent_sdk.config import (
    Config,
    get_image_steps,
    get_provider_config,
    get_tier_chain,
    load_config,
)


# ##################################################################
# default config — no file
# verify that load_config returns sensible defaults when no file exists
def test_default_config_no_file(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.yaml"
    cfg = load_config(missing)
    assert isinstance(cfg, Config)


# ##################################################################
# default tier chains
# verify all five tiers have non-empty chains with correct first entry
def test_default_tier_very_high(tmp_path: Path) -> None:
    cfg = load_config(tmp_path / "absent.yaml")
    chain = cfg.tiers["very_high"].chain
    assert len(chain) >= 1
    assert chain[0] == "claude:claude-opus-4-6"


def test_default_tier_high(tmp_path: Path) -> None:
    cfg = load_config(tmp_path / "absent.yaml")
    chain = cfg.tiers["high"].chain
    assert len(chain) >= 1
    assert chain[0] == "claude:claude-opus-4-6"


def test_default_tier_medium(tmp_path: Path) -> None:
    cfg = load_config(tmp_path / "absent.yaml")
    chain = cfg.tiers["medium"].chain
    assert chain[0] == "claude:claude-sonnet-4-6"


def test_default_tier_low(tmp_path: Path) -> None:
    cfg = load_config(tmp_path / "absent.yaml")
    chain = cfg.tiers["low"].chain
    assert chain[0] == "claude:claude-haiku-4-5-20251001"


def test_default_tier_free_fast(tmp_path: Path) -> None:
    cfg = load_config(tmp_path / "absent.yaml")
    chain = cfg.tiers["free_fast"].chain
    assert chain[0] == "ollama:qwen3-8b"


def test_default_tier_free_thinking(tmp_path: Path) -> None:
    cfg = load_config(tmp_path / "absent.yaml")
    chain = cfg.tiers["free_thinking"].chain
    assert chain[0] == "ollama:qwen3-30b-32k"


# ##################################################################
# default image config
# verify default model and step counts for each image tier
def test_default_image_model(tmp_path: Path) -> None:
    cfg = load_config(tmp_path / "absent.yaml")
    assert cfg.image.model == "z-image-turbo"


def test_default_image_steps_very_high(tmp_path: Path) -> None:
    cfg = load_config(tmp_path / "absent.yaml")
    assert cfg.image.tiers["very_high"].steps == 32


def test_default_image_steps_high(tmp_path: Path) -> None:
    cfg = load_config(tmp_path / "absent.yaml")
    assert cfg.image.tiers["high"].steps == 16


def test_default_image_steps_medium(tmp_path: Path) -> None:
    cfg = load_config(tmp_path / "absent.yaml")
    assert cfg.image.tiers["medium"].steps == 8


def test_default_image_steps_low(tmp_path: Path) -> None:
    cfg = load_config(tmp_path / "absent.yaml")
    assert cfg.image.tiers["low"].steps == 2


# ##################################################################
# default provider config
# verify default provider entries are present
def test_default_provider_claude(tmp_path: Path) -> None:
    cfg = load_config(tmp_path / "absent.yaml")
    assert "claude" in cfg.providers
    assert cfg.providers["claude"]["permission_mode"] == "bypassPermissions"


def test_default_provider_ollama(tmp_path: Path) -> None:
    cfg = load_config(tmp_path / "absent.yaml")
    assert cfg.providers["ollama"]["base_url"] == "http://localhost:11434"


def test_default_provider_no_vllm(tmp_path: Path) -> None:
    cfg = load_config(tmp_path / "absent.yaml")
    assert "vllm" not in cfg.providers


# ##################################################################
# default logging config
# verify default log directory, level and retention
def test_default_logging(tmp_path: Path) -> None:
    cfg = load_config(tmp_path / "absent.yaml")
    assert cfg.logging.level == "info"
    assert cfg.logging.retention_days == 30
    assert "logs" in cfg.logging.directory


# ##################################################################
# default fallback config
# verify default fallback strategies
def test_default_fallback_single_shot(tmp_path: Path) -> None:
    cfg = load_config(tmp_path / "absent.yaml")
    assert cfg.fallback.single_shot.strategy == "immediate_cascade"


def test_default_fallback_conversation(tmp_path: Path) -> None:
    cfg = load_config(tmp_path / "absent.yaml")
    assert cfg.fallback.conversation.strategy == "backoff_then_cascade"
    assert cfg.fallback.conversation.max_backoff_seconds == 60
    assert cfg.fallback.conversation.summarise_with == "free_thinking"


# ##################################################################
# custom config overrides
# write a real yaml file and verify overrides take effect
def test_custom_tier_chain(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    data = {
        "tiers": {
            "high": ["gemini:gemini-2.5-pro", "claude:claude-opus-4-6"],
        }
    }
    config_file.write_text(yaml.dump(data), encoding="utf-8")
    cfg = load_config(config_file)
    assert cfg.tiers["high"].chain == ["gemini:gemini-2.5-pro", "claude:claude-opus-4-6"]
    # other tiers fall back to defaults
    assert cfg.tiers["low"].chain[0] == "claude:claude-haiku-4-5-20251001"


def test_custom_image_model(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    data = {
        "image": {
            "model": "dall-e-3",
            "tiers": {
                "high": {"steps": 20},
                "medium": {"steps": 10},
                "low": {"steps": 2},
            },
        }
    }
    config_file.write_text(yaml.dump(data), encoding="utf-8")
    cfg = load_config(config_file)
    assert cfg.image.model == "dall-e-3"
    assert cfg.image.tiers["high"].steps == 20
    assert cfg.image.tiers["medium"].steps == 10
    assert cfg.image.tiers["low"].steps == 2


def test_custom_provider_config(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    data = {
        "providers": {
            "ollama": {"base_url": "http://10.0.0.5:11434"},
        }
    }
    config_file.write_text(yaml.dump(data), encoding="utf-8")
    cfg = load_config(config_file)
    assert cfg.providers["ollama"]["base_url"] == "http://10.0.0.5:11434"
    # non-overridden providers still get defaults
    assert cfg.providers["claude"]["permission_mode"] == "bypassPermissions"


def test_custom_logging_config(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    data = {
        "logging": {
            "level": "debug",
            "retention_days": 7,
            "directory": "/var/log/agent-sdk",
        }
    }
    config_file.write_text(yaml.dump(data), encoding="utf-8")
    cfg = load_config(config_file)
    assert cfg.logging.level == "debug"
    assert cfg.logging.retention_days == 7
    assert cfg.logging.directory == "/var/log/agent-sdk"


def test_custom_fallback_config(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    data = {
        "fallback": {
            "single_shot": {"strategy": "retry_then_cascade"},
            "conversation": {
                "strategy": "backoff_then_cascade",
                "max_backoff_seconds": 120,
                "summarise_with": "medium",
            },
        }
    }
    config_file.write_text(yaml.dump(data), encoding="utf-8")
    cfg = load_config(config_file)
    assert cfg.fallback.single_shot.strategy == "retry_then_cascade"
    assert cfg.fallback.conversation.max_backoff_seconds == 120
    assert cfg.fallback.conversation.summarise_with == "medium"


# ##################################################################
# get tier chain function
# verify correct chain returned for each tier enum value
def test_get_tier_chain_high(tmp_path: Path) -> None:
    cfg = load_config(tmp_path / "absent.yaml")
    chain = get_tier_chain(Tier.HIGH, cfg)
    assert chain[0] == "claude:claude-opus-4-6"
    assert len(chain) == 3


def test_get_tier_chain_medium(tmp_path: Path) -> None:
    cfg = load_config(tmp_path / "absent.yaml")
    chain = get_tier_chain(Tier.MEDIUM, cfg)
    assert chain[0] == "claude:claude-sonnet-4-6"


def test_get_tier_chain_low(tmp_path: Path) -> None:
    cfg = load_config(tmp_path / "absent.yaml")
    chain = get_tier_chain(Tier.LOW, cfg)
    assert chain[0] == "claude:claude-haiku-4-5-20251001"
    assert len(chain) == 3


def test_get_tier_chain_free_fast(tmp_path: Path) -> None:
    cfg = load_config(tmp_path / "absent.yaml")
    chain = get_tier_chain(Tier.FREE_FAST, cfg)
    assert chain[0] == "ollama:qwen3-8b"


def test_get_tier_chain_free_thinking(tmp_path: Path) -> None:
    cfg = load_config(tmp_path / "absent.yaml")
    chain = get_tier_chain(Tier.FREE_THINKING, cfg)
    assert chain[0] == "ollama:qwen3-30b-32k"


def test_get_tier_chain_returns_copy(tmp_path: Path) -> None:
    cfg = load_config(tmp_path / "absent.yaml")
    chain1 = get_tier_chain(Tier.HIGH, cfg)
    chain1.clear()
    chain2 = get_tier_chain(Tier.HIGH, cfg)
    assert len(chain2) > 0


# ##################################################################
# get image steps function
# verify correct step count for each tier
def test_get_image_steps_very_high(tmp_path: Path) -> None:
    cfg = load_config(tmp_path / "absent.yaml")
    assert get_image_steps(Tier.VERY_HIGH, cfg) == 32


def test_get_image_steps_high(tmp_path: Path) -> None:
    cfg = load_config(tmp_path / "absent.yaml")
    assert get_image_steps(Tier.HIGH, cfg) == 16


def test_get_image_steps_medium(tmp_path: Path) -> None:
    cfg = load_config(tmp_path / "absent.yaml")
    assert get_image_steps(Tier.MEDIUM, cfg) == 8


def test_get_image_steps_low(tmp_path: Path) -> None:
    cfg = load_config(tmp_path / "absent.yaml")
    assert get_image_steps(Tier.LOW, cfg) == 2


def test_get_image_steps_unknown_tier_returns_default(tmp_path: Path) -> None:
    cfg = load_config(tmp_path / "absent.yaml")
    # FREE_FAST has no image tier config — should return a sensible default
    steps = get_image_steps(Tier.FREE_FAST, cfg)
    assert isinstance(steps, int)
    assert steps > 0


# ##################################################################
# get provider config function
# verify correct dict returned for known and unknown providers
def test_get_provider_config_claude(tmp_path: Path) -> None:
    cfg = load_config(tmp_path / "absent.yaml")
    pc = get_provider_config("claude", cfg)
    assert pc["permission_mode"] == "bypassPermissions"


def test_get_provider_config_ollama(tmp_path: Path) -> None:
    cfg = load_config(tmp_path / "absent.yaml")
    pc = get_provider_config("ollama", cfg)
    assert pc["base_url"] == "http://localhost:11434"


def test_get_provider_config_gemini(tmp_path: Path) -> None:
    cfg = load_config(tmp_path / "absent.yaml")
    pc = get_provider_config("gemini", cfg)
    assert pc["api_key_env"] == "GEMINI_API_KEY"


def test_get_provider_config_unknown_returns_empty(tmp_path: Path) -> None:
    cfg = load_config(tmp_path / "absent.yaml")
    pc = get_provider_config("nonexistent_provider", cfg)
    assert pc == {}


def test_get_provider_config_returns_copy(tmp_path: Path) -> None:
    cfg = load_config(tmp_path / "absent.yaml")
    pc1 = get_provider_config("ollama", cfg)
    pc1["base_url"] = "mutated"
    pc2 = get_provider_config("ollama", cfg)
    assert pc2["base_url"] == "http://localhost:11434"


# ##################################################################
# invalid yaml handling
# verify that malformed yaml falls back to defaults rather than crashing
def test_invalid_yaml_uses_defaults(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("this: is: not: valid: yaml: :::::", encoding="utf-8")
    cfg = load_config(config_file)
    assert isinstance(cfg, Config)
    assert "high" in cfg.tiers
    assert cfg.image.model == "z-image-turbo"


def test_yaml_with_null_root_uses_defaults(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("null\n", encoding="utf-8")
    cfg = load_config(config_file)
    assert isinstance(cfg, Config)
    assert cfg.tiers["high"].chain[0] == "claude:claude-opus-4-6"


def test_yaml_with_empty_file_uses_defaults(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("", encoding="utf-8")
    cfg = load_config(config_file)
    assert isinstance(cfg, Config)
    assert "medium" in cfg.tiers


# ##################################################################
# caching behaviour
# verify load_config caches when no explicit path is given
def test_force_reload_returns_fresh_config(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    data1 = {"tiers": {"high": ["gemini:gemini-2.5-pro"]}}
    config_file.write_text(yaml.dump(data1), encoding="utf-8")
    cfg1 = load_config(config_file)
    assert cfg1.tiers["high"].chain == ["gemini:gemini-2.5-pro"]

    # modify file and reload with same explicit path — explicit path is never cached
    data2 = {"tiers": {"high": ["ollama:qwen3-8b"]}}
    config_file.write_text(yaml.dump(data2), encoding="utf-8")
    cfg2 = load_config(config_file)
    assert cfg2.tiers["high"].chain == ["ollama:qwen3-8b"]


# ##################################################################
# tts config
# verify default voices are populated
def test_default_tts_voices(tmp_path: Path) -> None:
    cfg = load_config(tmp_path / "absent.yaml")
    assert "gary" in cfg.tts.voices
    assert "aiden" in cfg.tts.voices
    assert cfg.tts.voices["gary"].provider == "local"


def test_default_tts_chain(tmp_path: Path) -> None:
    cfg = load_config(tmp_path / "absent.yaml")
    assert cfg.tts.default == ["local:qwen3-tts"]


def test_custom_tts_voices(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"
    data = {
        "tts": {
            "default": ["local:qwen3-tts"],
            "voices": {
                "alice": {"provider": "local", "voice_id": "alice", "description": "Friendly"},
            },
        }
    }
    config_file.write_text(yaml.dump(data), encoding="utf-8")
    cfg = load_config(config_file)
    assert "alice" in cfg.tts.voices
    assert cfg.tts.voices["alice"].description == "Friendly"
