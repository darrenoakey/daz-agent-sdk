from __future__ import annotations

import sys
import types

import pytest

from daz_agent_sdk.config import Config, TierConfig
from daz_agent_sdk.providers.base import Provider
from daz_agent_sdk.registry import (
    _PROVIDER_MODULES,
    _provider_cache,
    get_models_for_tier,
    get_provider,
    get_providers,
    refresh_providers,
    register_provider_module,
    resolve_model,
)
from daz_agent_sdk.types import Capability, ModelInfo, Tier


# ##################################################################
# helpers
# minimal in-memory provider for testing registry logic without
# importing real providers that may require optional dependencies


class _FakeProvider(Provider):
    name = "fake"

    async def available(self) -> bool:
        return True

    async def list_models(self) -> list[ModelInfo]:
        return _FAKE_MODELS

    async def complete(self, messages, model, **kwargs):
        raise NotImplementedError

    async def stream(self, messages, model, **kwargs):  # type: ignore[override]
        raise NotImplementedError
        yield  # make it an async generator


_FAKE_MODELS = [
    ModelInfo(
        provider="fake",
        model_id="fake-high",
        display_name="Fake High",
        capabilities=frozenset({Capability.TEXT, Capability.STRUCTURED}),
        tier=Tier.HIGH,
    ),
    ModelInfo(
        provider="fake",
        model_id="fake-low",
        display_name="Fake Low",
        capabilities=frozenset({Capability.TEXT}),
        tier=Tier.LOW,
    ),
]

# ##################################################################
# fake module
# synthetic module placed in sys.modules so register_provider_module
# can import it during tests


def _install_fake_module() -> None:
    mod = types.ModuleType("daz_agent_sdk.providers.fake")
    mod._FAKE_MODELS = _FAKE_MODELS  # type: ignore[attr-defined]

    class FakeProvider(_FakeProvider):
        pass

    FakeProvider.name = "fake"
    mod.FakeProvider = FakeProvider  # type: ignore[attr-defined]
    sys.modules["daz_agent_sdk.providers.fake"] = mod


def _remove_fake_module() -> None:
    sys.modules.pop("daz_agent_sdk.providers.fake", None)


# ##################################################################
# fixtures


@pytest.fixture(autouse=True)
def clean_registry():
    refresh_providers()
    _remove_fake_module()
    _PROVIDER_MODULES.pop("fake", None)
    yield
    refresh_providers()
    _remove_fake_module()
    _PROVIDER_MODULES.pop("fake", None)


# ##################################################################
# provider discovery tests


def test_get_providers_returns_dict():
    providers = get_providers()
    assert isinstance(providers, dict)


def test_get_providers_skips_missing_modules():
    register_provider_module("nonexistent_xyz", "daz_agent_sdk.providers.nonexistent_xyz")
    providers = get_providers()
    assert "nonexistent_xyz" not in providers


def test_get_providers_loads_installed_provider():
    _install_fake_module()
    register_provider_module("fake", "daz_agent_sdk.providers.fake")
    providers = get_providers()
    assert "fake" in providers
    assert isinstance(providers["fake"], Provider)


def test_get_providers_is_cached():
    _install_fake_module()
    register_provider_module("fake", "daz_agent_sdk.providers.fake")
    p1 = get_providers()
    p2 = get_providers()
    assert p1 == p2


def test_refresh_providers_clears_cache():
    _install_fake_module()
    register_provider_module("fake", "daz_agent_sdk.providers.fake")
    get_providers()
    assert "fake" in _provider_cache
    refresh_providers()
    assert len(_provider_cache) == 0


def test_get_provider_returns_none_for_missing():
    provider = get_provider("totally_nonexistent_provider")
    assert provider is None


def test_get_provider_loads_installed_provider():
    _install_fake_module()
    register_provider_module("fake", "daz_agent_sdk.providers.fake")
    provider = get_provider("fake")
    assert provider is not None
    assert isinstance(provider, Provider)


# ##################################################################
# tier chain resolution tests


def test_get_models_for_tier_returns_list():
    cfg = Config(tiers={"high": TierConfig(chain=["fake:fake-high"])})
    _install_fake_module()
    register_provider_module("fake", "daz_agent_sdk.providers.fake")
    models = get_models_for_tier(Tier.HIGH, config=cfg)
    assert isinstance(models, list)
    assert len(models) >= 1


def test_get_models_for_tier_respects_chain_order():
    cfg = Config(
        tiers={
            "high": TierConfig(chain=["fake:fake-high", "fake:fake-low"]),
        }
    )
    _install_fake_module()
    register_provider_module("fake", "daz_agent_sdk.providers.fake")
    models = get_models_for_tier(Tier.HIGH, config=cfg)
    assert models[0].model_id == "fake-high"
    assert models[1].model_id == "fake-low"


def test_get_models_for_tier_filters_by_capability():
    cfg = Config(
        tiers={
            "high": TierConfig(chain=["fake:fake-high", "fake:fake-low"]),
        }
    )
    _install_fake_module()
    register_provider_module("fake", "daz_agent_sdk.providers.fake")
    # only fake-high supports STRUCTURED
    models = get_models_for_tier(Tier.HIGH, capability=Capability.STRUCTURED, config=cfg)
    assert all(Capability.STRUCTURED in m.capabilities for m in models)
    assert any(m.model_id == "fake-high" for m in models)


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
    _install_fake_module()
    register_provider_module("fake", "daz_agent_sdk.providers.fake")
    info = resolve_model("fake", "fake-high", tier=Tier.HIGH)
    assert info is not None
    assert info.provider == "fake"
    assert info.model_id == "fake-high"
    assert info.display_name == "Fake High"


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
    _install_fake_module()
    register_provider_module("fake", "daz_agent_sdk.providers.fake")
    info1 = resolve_model("fake", "fake-high", tier=Tier.HIGH)
    info2 = resolve_model("fake", "fake-high", tier=Tier.HIGH)
    assert info1 is info2


def test_resolve_model_missing_provider_no_tier():
    result = resolve_model("notinstalled", "some-model")
    assert result is None


# ##################################################################
# register provider module tests


def test_register_provider_module_adds_to_known():
    register_provider_module("custom_provider", "some.module.path")
    assert "custom_provider" in _PROVIDER_MODULES
    assert _PROVIDER_MODULES["custom_provider"] == "some.module.path"
    _PROVIDER_MODULES.pop("custom_provider", None)
