from __future__ import annotations

import json
import socket
import subprocess
import threading
import urllib.request
from collections.abc import Iterator

import pytest


_ARBITER_HOST = "darren@10.0.0.254"
_ARBITER_REMOTE_PORT = 8400
_REQUIRED_ARBITER_MODELS = {"embed-text", "qwen3.6-27b", "qwen3.6-35b"}


# ##################################################################
# reserve loopback port
# ask the kernel for an unused local port immediately before ssh owns it
def _loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


# ##################################################################
# read arbiter models
# verify the exact tunnel endpoint and required live model registrations
def _read_arbiter_models(base_url: str) -> set[str]:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(f"{base_url}/v1/models", timeout=2.0) as response:
        if response.status != 200 or response.headers.get_content_type() != "application/json":
            raise RuntimeError(
                f"arbiter /v1/models verification failed: status={response.status} "
                f"content_type={response.headers.get_content_type()}"
            )
        payload = json.load(response)
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise RuntimeError("arbiter /v1/models did not return its exact JSON array schema")
    names = {
        str(item.get("llm_name") or item.get("model_id"))
        for item in payload
        if item.get("llm_name") or item.get("model_id")
    }
    missing = _REQUIRED_ARBITER_MODELS - names
    if missing:
        raise RuntimeError(f"arbiter /v1/models is missing required models: {sorted(missing)}")
    return names


# ##################################################################
# await tunnel readiness
# retry only while the exact ssh child remains alive, then fail loudly
def _await_tunnel(process: subprocess.Popen[bytes], base_url: str) -> None:
    last_error = "no request attempted"
    for _ in range(100):
        if process.poll() is not None:
            stderr = (process.stderr.read() if process.stderr is not None else b"").decode(
                errors="replace"
            )
            raise RuntimeError(
                f"arbiter ssh tunnel exited with code {process.returncode}: {stderr.strip()}"
            )
        try:
            _read_arbiter_models(base_url)
            return
        except Exception as exc:
            last_error = str(exc)
            threading.Event().wait(0.1)
    raise RuntimeError(f"arbiter ssh tunnel never passed /v1/models verification: {last_error}")


# ##################################################################
# stop tunnel
# terminate and reap only the ssh child owned by this pytest session
def _stop_tunnel(process: subprocess.Popen[bytes]) -> None:
    try:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5.0)
    finally:
        if process.stderr is not None:
            process.stderr.close()


# ##################################################################
# arbiter tunnel fixture
# provide every live Arbiter test one verified loopback-only real service route
@pytest.fixture(scope="session")
def arbiter_tunnel_url() -> Iterator[str]:
    port = _loopback_port()
    base_url = f"http://127.0.0.1:{port}"
    process = subprocess.Popen(
        [
            "/usr/bin/ssh",
            "-N",
            "-o",
            "BatchMode=yes",
            "-o",
            "ExitOnForwardFailure=yes",
            "-o",
            "ServerAliveInterval=15",
            "-o",
            "ServerAliveCountMax=2",
            "-L",
            f"127.0.0.1:{port}:127.0.0.1:{_ARBITER_REMOTE_PORT}",
            _ARBITER_HOST,
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        _await_tunnel(process, base_url)
        yield base_url
    finally:
        _stop_tunnel(process)
