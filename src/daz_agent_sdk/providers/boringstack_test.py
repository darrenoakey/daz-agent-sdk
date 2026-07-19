from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from daz_agent_sdk.providers.boringstack import BoringstackProvider
from daz_agent_sdk.providers.ollama import OllamaProvider
from daz_agent_sdk.registry import get_provider, refresh_providers, resolve_model
from daz_agent_sdk.types import Message, Tier


# ##################################################################
# boringstack host
# the dedicated remote Ollama box (Darren-Boringstack).
_BORINGSTACK_HOST = "10.0.0.42"
_BORINGSTACK_PORT = 11434


class _BoringstackProtocolHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path != "/api/tags":
            self.send_error(404)
            return
        self._send({"models": [{"name": "qwen3.6:35b-a3b"}]})

    def do_POST(self) -> None:
        if self.path != "/api/chat":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length))
        assert request["model"] == "qwen3.6:35b-a3b"
        self._send({"message": {"role": "assistant", "content": "pong"}})

    def _send(self, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


@pytest.fixture
def boringstack_endpoint() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _BoringstackProtocolHandler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        host = str(server.server_address[0])
        port = int(server.server_address[1])
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


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
# test protocol generation
# exercise the real Ollama wire protocol through a local TCP server.
@pytest.mark.asyncio
async def test_boringstack_protocol_generation(boringstack_endpoint: str) -> None:
    prov = BoringstackProvider(base_url=boringstack_endpoint)
    assert await prov.available()
    model = resolve_model("boringstack", "qwen3.6:35b-a3b", tier=Tier.FREE_FAST)
    assert model is not None
    messages = [
        Message(role="user", content="Reply with exactly the single word: pong")
    ]
    response = await prov.complete(messages, model)
    assert response.text.strip()
