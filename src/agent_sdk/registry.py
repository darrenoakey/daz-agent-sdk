from __future__ import annotations

import importlib
from agent_sdk.config import Config, get_tier_chain
from agent_sdk.providers.base import Provider
from agent_sdk.types import Capability, ModelInfo, Tier


# ##################################################################
# known providers
# list of provider names and their module paths for lazy import
_PROVIDER_MODULES: dict[str, str] = {
    "claude": "agent_sdk.providers.claude",
    "ollama": "agent_sdk.providers.ollama",
    "gemini": "agent_sdk.providers.gemini",
    "codex": "agent_sdk.providers.codex",
}

# ##################################################################
# provider cache
# lazily loaded provider instances, keyed by provider name
_provider_cache: dict[str, Provider] = {}

# ##################################################################
# model cache
# all ModelInfo objects from all loaded providers, keyed by qualified name
_model_cache: dict[str, ModelInfo] = {}


# ##################################################################
# load provider
# attempt to import a provider module by name — returns None if
# the module is not installed or fails to load. never raises.
def _load_provider(name: str) -> Provider | None:
    module_path = _PROVIDER_MODULES.get(name)
    if module_path is None:
        return None
    try:
        module = importlib.import_module(module_path)
        cls = getattr(module, f"{name.capitalize()}Provider", None)
        if cls is None:
            # try common naming patterns
            for attr_name in dir(module):
                obj = getattr(module, attr_name)
                if (
                    isinstance(obj, type)
                    and issubclass(obj, Provider)
                    and obj is not Provider
                    and getattr(obj, "name", None) == name
                ):
                    cls = obj
                    break
        if cls is None:
            return None
        instance = cls()
        return instance
    except Exception:
        return None


# ##################################################################
# register provider
# add a provider name and its module path to the known providers map
# useful for test injection and third-party provider plugins
def register_provider_module(name: str, module_path: str) -> None:
    _PROVIDER_MODULES[name] = module_path


# ##################################################################
# get providers
# returns all registered providers that can be loaded — providers
# whose module is missing or raises are silently skipped.
# result is cached after first load.
def get_providers() -> dict[str, Provider]:
    global _provider_cache
    if _provider_cache:
        return dict(_provider_cache)

    for name in list(_PROVIDER_MODULES.keys()):
        if name not in _provider_cache:
            provider = _load_provider(name)
            if provider is not None:
                _provider_cache[name] = provider

    return dict(_provider_cache)


# ##################################################################
# get provider
# returns a single provider by name, loading it lazily if needed.
# returns None if the provider is not installed or fails to load.
def get_provider(name: str) -> Provider | None:
    if name in _provider_cache:
        return _provider_cache[name]
    provider = _load_provider(name)
    if provider is not None:
        _provider_cache[name] = provider
    return provider


# ##################################################################
# refresh providers
# clears the provider and model caches, forcing reload on next access.
# used in tests and when provider availability may have changed.
# uses .clear() to preserve the same dict object so imported references
# in test modules stay valid after a refresh.
def refresh_providers() -> None:
    _provider_cache.clear()
    _model_cache.clear()


# ##################################################################
# get models for tier
# returns the ordered list of ModelInfo objects for a tier, resolved
# from the config chain. if capability is provided, only models that
# include that capability are returned.
# providers that are not installed produce placeholder ModelInfo entries
# so the tier chain is preserved for fallback ordering.
def get_models_for_tier(
    tier: Tier,
    capability: Capability | None = None,
    config: Config | None = None,
) -> list[ModelInfo]:
    chain = get_tier_chain(tier, config)
    results: list[ModelInfo] = []

    for entry in chain:
        if ":" not in entry:
            continue
        provider_name, model_id = entry.split(":", 1)
        info = resolve_model(provider_name, model_id, tier=tier)
        if info is None:
            continue
        if capability is not None and capability not in info.capabilities:
            continue
        results.append(info)

    return results


# ##################################################################
# resolve model
# look up a ModelInfo by provider name and model id.
# checks the loaded provider's model list first. if the provider is
# not loaded, constructs a minimal ModelInfo from the known names.
def resolve_model(
    provider_name: str,
    model_id: str,
    *,
    tier: Tier | None = None,
) -> ModelInfo | None:
    qualified = f"{provider_name}:{model_id}"

    if qualified in _model_cache:
        return _model_cache[qualified]

    provider = get_provider(provider_name)
    if provider is None:
        if tier is None:
            return None
        # build a minimal placeholder so the chain is preserved
        info = ModelInfo(
            provider=provider_name,
            model_id=model_id,
            display_name=f"{provider_name}/{model_id}",
            capabilities=frozenset({Capability.TEXT}),
            tier=tier,
        )
        _model_cache[qualified] = info
        return info

    # scan the provider's static model list without making async calls
    models = _get_static_models(provider)
    for m in models:
        _model_cache[f"{m.provider}:{m.model_id}"] = m

    if qualified in _model_cache:
        return _model_cache[qualified]

    if tier is None:
        return None

    # model not in provider's static list — build placeholder
    info = ModelInfo(
        provider=provider_name,
        model_id=model_id,
        display_name=f"{provider_name}/{model_id}",
        capabilities=frozenset({Capability.TEXT}),
        tier=tier,
    )
    _model_cache[qualified] = info
    return info


# ##################################################################
# get static models
# retrieve the static _MODELS or similar attribute from a provider
# without making async calls. returns an empty list if not found.
def _get_static_models(provider: Provider) -> list[ModelInfo]:
    module = type(provider).__module__
    try:
        mod = importlib.import_module(module)
    except Exception:
        return []

    # look for the conventional _<PROVIDER>_MODELS list
    provider_name = getattr(provider, "name", "").upper()
    candidate_names = [
        f"_{provider_name}_MODELS",
        "_MODELS",
    ]
    for attr in candidate_names:
        models = getattr(mod, attr, None)
        if isinstance(models, list):
            return [m for m in models if isinstance(m, ModelInfo)]

    return []
