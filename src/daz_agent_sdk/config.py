from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from daz_agent_sdk.types import Tier


# default config path
_CONFIG_PATH = Path.home() / ".daz-agent-sdk" / "config.yaml"

# module-level cache — loaded once per process
_config_cache: Config | None = None


# ##################################################################
# tier config
# ordered list of provider:model strings for a tier
@dataclass
class TierConfig:
    chain: list[str] = field(default_factory=list)


# ##################################################################
# image tier config
# step count for one image quality tier
@dataclass
class ImageTierConfig:
    steps: int = 4


# ##################################################################
# image config
# model selection and per-tier step counts for image generation
@dataclass
class ImageConfig:
    model: str = "z-image-turbo"
    tiers: dict[str, ImageTierConfig] = field(default_factory=dict)
    fallback: list[str] = field(default_factory=list)
    transparent_post_process: str = "birefnet"

    # ##################################################################
    # post init
    # fill default tier step counts if not provided
    def __post_init__(self) -> None:
        defaults = {
            "very_high": ImageTierConfig(steps=32),
            "high": ImageTierConfig(steps=16),
            "medium": ImageTierConfig(steps=8),
            "low": ImageTierConfig(steps=2),
        }
        for tier_key, default in defaults.items():
            if tier_key not in self.tiers:
                self.tiers[tier_key] = default


# ##################################################################
# tts voice config
# per-voice provider and id settings
@dataclass
class TtsVoiceConfig:
    provider: str = "local"
    voice_id: str = ""
    description: str = ""


# ##################################################################
# tts config
# default provider chain and named voice definitions
@dataclass
class TtsConfig:
    default: list[str] = field(default_factory=lambda: ["local:qwen3-tts"])
    voices: dict[str, TtsVoiceConfig] = field(default_factory=dict)

    # ##################################################################
    # post init
    # fill default voices if not provided
    def __post_init__(self) -> None:
        if not self.voices:
            self.voices = {
                "gary": TtsVoiceConfig(provider="local", voice_id="gary", description="British newsreader"),
                "aiden": TtsVoiceConfig(provider="local", voice_id="aiden"),
            }


# ##################################################################
# logging config
# directory, level and retention for conversation logs
@dataclass
class LoggingConfig:
    directory: str = "~/.daz-agent-sdk/logs"
    level: str = "info"
    retention_days: int = 30


# ##################################################################
# fallback single shot config
# strategy for one-shot requests
@dataclass
class FallbackSingleShotConfig:
    strategy: str = "immediate_cascade"


# ##################################################################
# fallback conversation config
# strategy for multi-turn conversations including backoff and cascade
@dataclass
class FallbackConversationConfig:
    strategy: str = "backoff_then_cascade"
    max_backoff_seconds: int = 60
    summarise_with: str = "free_thinking"


# ##################################################################
# fallback config
# top-level fallback settings for single-shot and conversation modes
@dataclass
class FallbackConfig:
    single_shot: FallbackSingleShotConfig = field(default_factory=FallbackSingleShotConfig)
    conversation: FallbackConversationConfig = field(default_factory=FallbackConversationConfig)


# ##################################################################
# config
# top-level configuration object loaded from ~/.daz-agent-sdk/config.yaml
# all fields have sensible defaults so no config file is required
@dataclass
class Config:
    tiers: dict[str, TierConfig] = field(default_factory=dict)
    providers: dict[str, dict[str, Any]] = field(default_factory=dict)
    image: ImageConfig = field(default_factory=ImageConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    fallback: FallbackConfig = field(default_factory=FallbackConfig)

    # ##################################################################
    # post init
    # fill default tier chains if not provided by config file
    def __post_init__(self) -> None:
        defaults: dict[str, list[str]] = {
            "very_high": [
                "claude:claude-opus-4-6",
                "codex:gpt-5.3-codex",
                "gemini:gemini-2.5-pro",
            ],
            "high": [
                "claude:claude-opus-4-6",
                "codex:gpt-5.3-codex",
                "gemini:gemini-2.5-pro",
            ],
            "medium": [
                "claude:claude-sonnet-4-6",
                "codex:gpt-5.3-codex",
                "gemini:gemini-2.5-flash",
            ],
            "low": [
                "claude:claude-haiku-4-5-20251001",
                "gemini:gemini-2.5-flash-lite",
                "ollama:qwen3-8b",
            ],
            "free_fast": [
                "ollama:qwen3-8b",
            ],
            "free_thinking": [
                "ollama:qwen3-30b-32k",
                "ollama:deepseek-r1:14b",
            ],
        }
        for tier_key, chain in defaults.items():
            if tier_key not in self.tiers:
                self.tiers[tier_key] = TierConfig(chain=chain)

        default_providers: dict[str, dict[str, Any]] = {
            "claude": {"permission_mode": "bypassPermissions"},
            "codex": {},
            "gemini": {"api_key_env": "GEMINI_API_KEY"},
            "ollama": {"base_url": "http://localhost:11434"},
        }
        for provider_key, provider_cfg in default_providers.items():
            if provider_key not in self.providers:
                self.providers[provider_key] = provider_cfg


# ##################################################################
# build image config from raw dict
# converts the yaml image section into an ImageConfig dataclass
def _build_image_config(raw: dict[str, Any]) -> ImageConfig:
    tiers: dict[str, ImageTierConfig] = {}
    raw_tiers = raw.get("tiers") or {}
    for tier_key, tier_val in raw_tiers.items():
        steps = tier_val.get("steps", 4) if isinstance(tier_val, dict) else 4
        tiers[tier_key] = ImageTierConfig(steps=steps)

    fallback = raw.get("fallback") or []
    transparent = raw.get("transparent") or {}
    post_process = transparent.get("post_process", "birefnet") if isinstance(transparent, dict) else "birefnet"

    return ImageConfig(
        model=raw.get("model", "z-image-turbo"),
        tiers=tiers,
        fallback=list(fallback),
        transparent_post_process=post_process,
    )


# ##################################################################
# build tts config from raw dict
# converts the yaml tts section into a TtsConfig dataclass
def _build_tts_config(raw: dict[str, Any]) -> TtsConfig:
    default_chain = raw.get("default") or ["local:qwen3-tts"]
    voices: dict[str, TtsVoiceConfig] = {}
    raw_voices = raw.get("voices") or {}
    for voice_name, voice_val in raw_voices.items():
        if isinstance(voice_val, dict):
            voices[voice_name] = TtsVoiceConfig(
                provider=voice_val.get("provider", "local"),
                voice_id=voice_val.get("voice_id", voice_name),
                description=voice_val.get("description", ""),
            )

    tts = TtsConfig(default=list(default_chain), voices=voices)
    # skip __post_init__ default filling since we supplied voices from config
    return tts


# ##################################################################
# build fallback config from raw dict
# converts the yaml fallback section into a FallbackConfig dataclass
def _build_fallback_config(raw: dict[str, Any]) -> FallbackConfig:
    ss_raw = raw.get("single_shot") or {}
    conv_raw = raw.get("conversation") or {}

    single_shot = FallbackSingleShotConfig(
        strategy=ss_raw.get("strategy", "immediate_cascade") if isinstance(ss_raw, dict) else "immediate_cascade",
    )
    conversation = FallbackConversationConfig(
        strategy=conv_raw.get("strategy", "backoff_then_cascade") if isinstance(conv_raw, dict) else "backoff_then_cascade",
        max_backoff_seconds=conv_raw.get("max_backoff_seconds", 60) if isinstance(conv_raw, dict) else 60,
        summarise_with=conv_raw.get("summarise_with", "free_thinking") if isinstance(conv_raw, dict) else "free_thinking",
    )
    return FallbackConfig(single_shot=single_shot, conversation=conversation)


# ##################################################################
# parse raw yaml dict into Config
# extracts all sections, falling back to defaults for missing keys
def _parse_raw(raw: dict[str, Any]) -> Config:
    # tiers
    tiers: dict[str, TierConfig] = {}
    raw_tiers = raw.get("tiers") or {}
    for tier_key, chain in raw_tiers.items():
        tiers[tier_key] = TierConfig(chain=list(chain) if chain else [])

    # providers
    providers: dict[str, dict[str, Any]] = {}
    raw_providers = raw.get("providers") or {}
    for provider_key, provider_val in raw_providers.items():
        providers[provider_key] = dict(provider_val) if isinstance(provider_val, dict) else {}

    # image
    raw_image = raw.get("image") or {}
    image = _build_image_config(raw_image) if isinstance(raw_image, dict) else ImageConfig()

    # tts
    raw_tts = raw.get("tts") or {}
    tts = _build_tts_config(raw_tts) if isinstance(raw_tts, dict) else TtsConfig()

    # logging
    raw_logging = raw.get("logging") or {}
    logging_cfg = LoggingConfig(
        directory=raw_logging.get("directory", "~/.daz-agent-sdk/logs") if isinstance(raw_logging, dict) else "~/.daz-agent-sdk/logs",
        level=raw_logging.get("level", "info") if isinstance(raw_logging, dict) else "info",
        retention_days=raw_logging.get("retention_days", 30) if isinstance(raw_logging, dict) else 30,
    )

    # fallback
    raw_fallback = raw.get("fallback") or {}
    fallback = _build_fallback_config(raw_fallback) if isinstance(raw_fallback, dict) else FallbackConfig()

    return Config(
        tiers=tiers,
        providers=providers,
        image=image,
        tts=tts,
        logging=logging_cfg,
        fallback=fallback,
    )


# ##################################################################
# load config
# reads ~/.daz-agent-sdk/config.yaml if it exists, otherwise uses defaults
# result is cached at module level for the lifetime of the process
def load_config(config_path: Path | None = None, *, force_reload: bool = False) -> Config:
    global _config_cache
    if _config_cache is not None and not force_reload and config_path is None:
        return _config_cache

    path = config_path or _CONFIG_PATH

    if not path.exists():
        cfg = Config()
        if config_path is None:
            _config_cache = cfg
        return cfg

    try:
        text = path.read_text(encoding="utf-8")
        raw = yaml.safe_load(text)
        if not isinstance(raw, dict):
            cfg = Config()
        else:
            cfg = _parse_raw(raw)
    except (yaml.YAMLError, OSError):
        cfg = Config()

    if config_path is None:
        _config_cache = cfg

    return cfg


# ##################################################################
# get tier chain
# returns the ordered list of "provider:model" strings for a given tier
# falls back to the default config if the tier has no chain defined
def get_tier_chain(tier: Tier, config: Config | None = None) -> list[str]:
    cfg = config or load_config()
    tier_cfg = cfg.tiers.get(tier.value)
    if tier_cfg is None:
        return []
    return list(tier_cfg.chain)


# ##################################################################
# get image steps
# returns the inference step count for a given image generation tier
# only high/medium/low are meaningful for image generation
def get_image_steps(tier: Tier, config: Config | None = None) -> int:
    cfg = config or load_config()
    tier_cfg = cfg.image.tiers.get(tier.value)
    if tier_cfg is None:
        return cfg.image.tiers.get("medium", ImageTierConfig(steps=4)).steps
    return tier_cfg.steps


# ##################################################################
# get provider config
# returns the provider-specific config dict for a named provider
# returns an empty dict if the provider is not configured
def get_provider_config(provider: str, config: Config | None = None) -> dict[str, Any]:
    cfg = config or load_config()
    return dict(cfg.providers.get(provider, {}))
