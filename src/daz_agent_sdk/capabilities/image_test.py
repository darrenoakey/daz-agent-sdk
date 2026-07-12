from __future__ import annotations

import asyncio
import base64
import json
import shutil
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

import daz_agent_sdk.capabilities.image as image_module
from daz_agent_sdk.capabilities.image import generate_image
from daz_agent_sdk.types import AgentError, ErrorKind


# ##################################################################
# helpers


def _codex_available() -> bool:
    """codex CLI on PATH is a proxy for 'this machine is set up for image gen'."""
    return shutil.which("codex") is not None


def _service_reachable() -> bool:
    """Check whether the mac mini image_generation_service answers at /jobs."""
    from urllib.request import urlopen
    from urllib.error import URLError
    try:
        with urlopen(image_module._IMAGE_SERVICE_URL + "/jobs", timeout=3):
            return True
    except (URLError, OSError):
        # A 404/405 still means the service is up — only connection errors count.
        try:
            with urlopen(image_module._IMAGE_SERVICE_URL, timeout=3):
                return True
        except (URLError, OSError):
            return False
        except Exception:
            return True


# ##################################################################
# provider validation


def test_rejects_unknown_provider():
    """Only 'codex' is supported now — explicit requests for other providers
    (legacy spark/mflux/nano-banana-2) fail loudly rather than silently rerouting."""
    with pytest.raises(AgentError) as exc_info:
        asyncio.run(
            generate_image(
                "test",
                width=512,
                height=512,
                provider="spark",
                timeout=1.0,
            )
        )
    assert "spark" in str(exc_info.value)
    assert exc_info.value.kind == ErrorKind.INVALID_REQUEST


def test_codex_is_the_default_provider(tmp_path, monkeypatch):
    """When no provider is requested, the request is dispatched as codex."""
    captured: dict[str, object] = {}
    png_data = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNgYPgPAAEDAQCR"
        "9I9ZAAAAAElFTkSuQmCC"
    )

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def do_POST(self):  # noqa: N802
            length = int(self.headers["Content-Length"])
            captured["body"] = json.loads(self.rfile.read(length))
            self.send_response(202)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"id":"job-1"}')

        def do_GET(self):  # noqa: N802
            if self.path == "/jobs/job-1":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"id":"job-1","status":"done","attempts":1}')
                return
            if self.path == "/jobs/job-1/image":
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.end_headers()
                self.wfile.write(png_data)
                return
            self.send_response(404)
            self.end_headers()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        monkeypatch.setattr(
            image_module, "_IMAGE_SERVICE_URL",
            f"http://127.0.0.1:{server.server_port}",
        )
        result = asyncio.run(
            generate_image(
                "a flat blue square",
                width=512,
                height=512,
                output=str(tmp_path / "out.png"),
                timeout=10.0,
            )
        )
    finally:
        server.shutdown()
        server.server_close()

    assert result.model_used.provider == "codex"
    assert result.model_used.model_id == "macmini-image-service"


# ##################################################################
# image service job submission


def test_codex_submits_input_image_to_service(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """codex provider uploads input images to the shared image service."""
    png_data = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNgYPgPAAEDAQCR"
        "9I9ZAAAAAElFTkSuQmCC"
    )
    input_path = tmp_path / "source.png"
    input_path.write_bytes(png_data)
    captured: dict[str, object] = {}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def do_POST(self):  # noqa: N802
            assert self.path == "/jobs"
            length = int(self.headers["Content-Length"])
            captured["body"] = json.loads(self.rfile.read(length))
            self.send_response(202)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"id":"job-1"}')

        def do_GET(self):  # noqa: N802
            if self.path == "/jobs/job-1":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"id":"job-1","status":"done","attempts":1}')
                return
            if self.path == "/jobs/job-1/image":
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.end_headers()
                self.wfile.write(png_data)
                return
            self.send_response(404)
            self.end_headers()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        monkeypatch.setattr(
            image_module, "_IMAGE_SERVICE_URL",
            f"http://127.0.0.1:{server.server_port}",
        )
        result = asyncio.run(
            generate_image(
                "edit this",
                width=512,
                height=512,
                image=input_path,
                provider="codex",
                timeout=30.0,
            )
        )
    finally:
        server.shutdown()
        server.server_close()

    assert result.model_used.model_id == "macmini-image-service"
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["prompt"] == "edit this"
    assert body["width"] == 512
    assert body["height"] == 512
    assert body["source_images"] == [base64.b64encode(png_data).decode("ascii")]


def test_input_image_missing_raises():
    """A non-existent input image is rejected before the service is called."""
    with pytest.raises(AgentError) as exc_info:
        asyncio.run(
            generate_image(
                "edit this",
                width=512,
                height=512,
                image="/nonexistent/does-not-exist.png",
                timeout=1.0,
            )
        )
    assert exc_info.value.kind == ErrorKind.INVALID_REQUEST


# ##################################################################
# real end-to-end test — only runs when the service is reachable


def test_generate_real_image():
    """Generate a real 512x512 image via the mac mini image service and verify
    it's a valid PNG. Skipped when the service is unreachable."""
    if not _service_reachable():
        pytest.skip("mac mini image_generation_service not reachable at :8830")

    result = asyncio.run(
        generate_image(
            "A cheerful cartoon robot waving hello",
            width=512,
            height=512,
            timeout=600.0,
        )
    )

    assert result.path.exists(), f"Image file not created: {result.path}"
    assert result.path.stat().st_size > 1000, "Image file suspiciously small"
    assert result.width == 512
    assert result.height == 512
    assert result.model_used.model_id == "macmini-image-service"

    header = result.path.read_bytes()[:8]
    is_png = header[:4] == b"\x89PNG"
    is_jpeg = header[:2] == b"\xff\xd8"
    assert is_png or is_jpeg, f"Not a valid image, header: {header!r}"
    print(f"\nGenerated image at: {result.path}")
    print(f"File size: {result.path.stat().st_size} bytes")
