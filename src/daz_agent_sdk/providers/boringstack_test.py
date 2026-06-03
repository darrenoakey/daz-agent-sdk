from __future__ import annotations

import socket

import pytest
import pytest_asyncio  # noqa: F401 — registers asyncio mode

from daz_agent_sdk.providers.boringstack import BoringstackProvider
from daz_agent_sdk.providers.ollama import OllamaProvider
from daz_agent_sdk.registry import get_provider, refresh_providers, resolve_model
from daz_agent_sdk.types import Message, Tier


# ##################################################################
# boringstack host
# the dedicated remote Ollama box (Darren-Boringstack).
_BORINGSTACK_HOST = "10.0.0.237"
_BORINGSTACK_PORT = 11434


# ##################################################################
# boringstack reachability check
# probe at module level so the live generation test skips quickly when
# the boringstack box is offline, rather than timing out per-test.
def _boringstack_reachable() -> bool:
    try:
        s = socket.create_connection((_BORINGSTACK_HOST, _BORINGSTACK_PORT), timeout=2)
        s.close()
        return True
    except OSError:
        return False


BORINGSTACK_RUNNING = _boringstack_reachable()
skip_if_offline = pytest.mark.skipif(
    not BORINGSTACK_RUNNING,
    reason=f"boringstack not reachable at {_BORINGSTACK_HOST}:{_BORINGSTACK_PORT}",
)


# ##################################################################
# test identity
# boringstack is an Ollama provider under a distinct name pointed at the
# remote host by default.
def test_boringstack_identity() -> None:
    prov = BoringstackProvider()
    assert prov.name == "boringstack"
    assert isinstance(prov, OllamaProvider)
    assert prov._base_url == f"http://{_BORINGSTACK_HOST}:{_BORINGSTACK_PORT}"


# ##################################################################
# test base url override
# an explicit base_url (as the registry passes from config) wins over the
# remote default.
def test_boringstack_base_url_override() -> None:
    prov = BoringstackProvider(base_url="http://example.local:1234/")
    assert prov._base_url == "http://example.local:1234"


# ##################################################################
# test registry resolution
# the registry must resolve "boringstack" to a BoringstackProvider so that
# tier chains like "boringstack:qwen3.6:35b-a3b" work.
def test_boringstack_in_registry() -> None:
    refresh_providers()
    prov = get_provider("boringstack")
    assert isinstance(prov, BoringstackProvider)


# ##################################################################
# test live generation
# real call against the qwen3.6:35b-a3b MoE on boringstack — no mocks.
# skipped when the box is offline.
@skip_if_offline
@pytest.mark.asyncio
async def test_boringstack_live_generation() -> None:
    prov = BoringstackProvider()
    assert await prov.available()
    model = resolve_model("boringstack", "qwen3.6:35b-a3b", tier=Tier.FREE_FAST)
    assert model is not None
    messages = [Message(role="user", content="Reply with exactly the single word: pong")]
    response = await prov.complete(messages, model)
    assert response.text.strip()
