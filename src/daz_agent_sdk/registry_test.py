from __future__ import annotations

from pathlib import Path
from typing import AsyncIterator, Type
from uuid import uuid4

import pytest

from daz_agent_sdk.config import Config, TierConfig
from daz_agent_sdk.providers.base import Provider
from daz_agent_sdk.registry import (
    _PROVIDER_MODULES,
    _load_provider,
    _provider_cache,
    get_models_for_tier,
    get_provider,
    get_providers,
    refresh_providers,
    register_provider_module,
    resolve_model,
)
from daz_agent_sdk.types import (
    Capability,
    Message,
    ModelInfo,
    Response,
    StructuredResponse,
    T,
    Tier,
)


# ##################################################################
# helpers
# concrete in-process provider for exercising registry behavior


class RegistryProvider(Provider):
    name = "registry"

    async def available(self) -> bool:
        return True

    async def list_models(self) -> list[ModelInfo]:
        return _REGISTRY_MODELS

    async def complete(
        self,
        messages: list[Message],
        model: ModelInfo,
        *,
        schema: Type[T] | None = None,
        tools: list[str] | None = None,
        cwd: str | Path | None = None,
        max_turns: int = 1,
        max_tokens: int | None = None,
        timeout: float = 300.0,
        setting_sources: list[str] | tuple[str, ...] | None = None,
    ) -> Response | StructuredResponse:
        text = messages[-1].content if messages else ""
        return Response(
            text=text,
            model_used=model,
            conversation_id=uuid4(),
            turn_id=uuid4(),
        )

    async def stream(
        self,
        messages: list[Message],
        model: ModelInfo,
        *,
        timeout: float = 300.0,
    ) -> AsyncIterator[str]:
        response = await self.complete(messages, model, timeout=timeout)
        yield response.text


_REGISTRY_MODELS = [
    ModelInfo(
        provider="registry",
        model_id="registry-high",
        display_name="Registry High",
        capabilities=frozenset({Capability.TEXT, Capability.STRUCTURED}),
        tier=Tier.HIGH,
    ),
    ModelInfo(
        provider="registry",
        model_id="registry-low",
        display_name="Registry Low",
        capabilities=frozenset({Capability.TEXT}),
        tier=Tier.LOW,
    ),
]

# ##################################################################
# fixtures


@pytest.fixture(autouse=True)
def clean_registry():
    refresh_providers()
    _PROVIDER_MODULES.pop("registry", None)
    yield
    refresh_providers()
    _PROVIDER_MODULES.pop("registry", None)


# ##################################################################
# provider discovery tests


def test_get_providers_returns_dict():
    providers = get_providers()
    assert isinstance(providers, dict)


def test_get_providers_skips_missing_modules():
    register_provider_module(
        "nonexistent_xyz", "daz_agent_sdk.providers.nonexistent_xyz"
    )
    providers = get_providers()
    assert "nonexistent_xyz" not in providers


def test_get_providers_loads_installed_provider():
    register_provider_module("registry", "daz_agent_sdk.registry_test")
    providers = get_providers()
    assert "registry" in providers
    assert isinstance(providers["registry"], Provider)


def test_get_providers_is_cached():
    register_provider_module("registry", "daz_agent_sdk.registry_test")
    p1 = get_providers()
    p2 = get_providers()
    assert p1 == p2


def test_refresh_providers_clears_cache():
    register_provider_module("registry", "daz_agent_sdk.registry_test")
    get_providers()
    assert "registry" in _provider_cache
    refresh_providers()
    assert len(_provider_cache) == 0


def test_get_provider_returns_none_for_missing():
    provider = get_provider("totally_nonexistent_provider")
    assert provider is None


def test_get_provider_loads_installed_provider():
    register_provider_module("registry", "daz_agent_sdk.registry_test")
    provider = get_provider("registry")
    assert provider is not None
    assert isinstance(provider, Provider)


# ##################################################################
# tier chain resolution tests


def test_get_models_for_tier_returns_list():
    cfg = Config(tiers={"high": TierConfig(chain=["registry:registry-high"])})
    register_provider_module("registry", "daz_agent_sdk.registry_test")
    models = get_models_for_tier(Tier.HIGH, config=cfg)
    assert isinstance(models, list)
    assert len(models) >= 1


def test_get_models_for_tier_respects_chain_order():
    cfg = Config(
        tiers={
            "high": TierConfig(chain=["registry:registry-high", "registry:registry-low"]),
        }
    )
    register_provider_module("registry", "daz_agent_sdk.registry_test")
    models = get_models_for_tier(Tier.HIGH, config=cfg)
    assert models[0].model_id == "registry-high"
    assert models[1].model_id == "registry-low"


def test_get_models_for_tier_filters_by_capability():
    cfg = Config(
        tiers={
            "high": TierConfig(chain=["registry:registry-high", "registry:registry-low"]),
        }
    )
    register_provider_module("registry", "daz_agent_sdk.registry_test")
    # only registry-high supports STRUCTURED
    models = get_models_for_tier(
        Tier.HIGH, capability=Capability.STRUCTURED, config=cfg
    )
    assert all(Capability.STRUCTURED in m.capabilities for m in models)
    assert any(m.model_id == "registry-high" for m in models)


def test_get_models_for_tier_empty_chain():
    cfg = Config(tiers={"high": TierConfig(chain=[])})
    models = get_models_for_tier(Tier.HIGH, config=cfg)
    assert models == []


def test_get_models_for_tier_provider_not_installed():
    cfg = Config(tiers={"high": TierConfig(chain=["ghost_provider:ghost-model"])})
    models = get_models_for_tier(Tier.HIGH, config=cfg)
    # placeholder ModelInfo should still be returned for ordering
    assert len(models) == 1
    assert models[0].provider == "ghost_provider"
    assert models[0].model_id == "ghost-model"


# ##################################################################
# model lookup tests


def test_resolve_model_finds_model_in_provider():
    register_provider_module("registry", "daz_agent_sdk.registry_test")
    info = resolve_model("registry", "registry-high", tier=Tier.HIGH)
    assert info is not None
    assert info.provider == "registry"
    assert info.model_id == "registry-high"
    assert info.display_name == "Registry High"


def test_resolve_model_returns_none_without_tier_for_missing():
    info = resolve_model("ghost_provider", "ghost-model")
    assert info is None


def test_resolve_model_returns_placeholder_with_tier_for_missing():
    info = resolve_model("ghost_provider", "ghost-model", tier=Tier.LOW)
    assert info is not None
    assert info.provider == "ghost_provider"
    assert info.model_id == "ghost-model"
    assert info.tier == Tier.LOW


def test_resolve_model_caches_result():
    register_provider_module("registry", "daz_agent_sdk.registry_test")
    info1 = resolve_model("registry", "registry-high", tier=Tier.HIGH)
    info2 = resolve_model("registry", "registry-high", tier=Tier.HIGH)
    assert info1 is info2


def test_resolve_model_missing_provider_no_tier():
    result = resolve_model("notinstalled", "some-model")
    assert result is None


# ##################################################################
# provider base_url from config tests


def test_load_provider_passes_base_url_from_config():
    config = Config(providers={"boringstack": {"base_url": "http://custom-host:9999"}})
    provider = _load_provider("boringstack", config)
    assert provider is not None
    assert getattr(provider, "_base_url") == "http://custom-host:9999"


# ##################################################################
# register provider module tests


def test_register_provider_module_adds_to_known():
    register_provider_module("custom_provider", "some.module.path")
    assert "custom_provider" in _PROVIDER_MODULES
    assert _PROVIDER_MODULES["custom_provider"] == "some.module.path"
    _PROVIDER_MODULES.pop("custom_provider", None)
