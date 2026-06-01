from __future__ import annotations

import asyncio

import pytest

from daz_agent_sdk.capabilities.image import (
    _closest_aspect_ratio,
    _image_size,
    generate_image,
)
from daz_agent_sdk.types import AgentError, ErrorKind


# ##################################################################
# aspect ratio tests


def test_square_aspect_ratio():
    assert _closest_aspect_ratio(512, 512) == "1:1"


def test_landscape_aspect_ratio():
    assert _closest_aspect_ratio(1920, 1080) == "16:9"


def test_portrait_aspect_ratio():
    assert _closest_aspect_ratio(1080, 1920) == "9:16"


def test_4_3_aspect_ratio():
    assert _closest_aspect_ratio(800, 600) == "4:3"


def test_3_4_aspect_ratio():
    assert _closest_aspect_ratio(600, 800) == "3:4"


# ##################################################################
# image size tests


def test_image_size_small():
    assert _image_size(512, 512) == "0.5K"


def test_image_size_1k():
    assert _image_size(1024, 1024) == "1K"


def test_image_size_1k_non_square():
    assert _image_size(1024, 768) == "1K"


def test_image_size_2k():
    assert _image_size(2048, 1024) == "2K"


def test_image_size_4k():
    assert _image_size(3840, 2160) == "4K"


def test_image_size_tiny_maps_to_half_k():
    assert _image_size(256, 256) == "0.5K"


# ##################################################################
# real image generation test — calls Nano Banana 2 API


def test_generate_real_image():
    """Generate a real 512x512 image via Nano Banana 2 and verify it's a valid PNG."""
    result = asyncio.run(
        generate_image(
            "A cheerful cartoon robot waving hello",
            width=512,
            height=512,
            timeout=60.0,
        )
    )

    assert result.path.exists(), f"Image file not created: {result.path}"
    assert result.path.stat().st_size > 1000, "Image file suspiciously small"
    assert result.width == 512
    assert result.height == 512
    assert result.model_used.model_id, "model_used.model_id should not be empty"

    header = result.path.read_bytes()[:8]
    is_png = header[:4] == b"\x89PNG"
    is_jpeg = header[:2] == b"\xff\xd8"
    assert is_png or is_jpeg, f"Not a valid image, header: {header!r}"
    print(f"\nGenerated image at: {result.path}")
    print(f"File size: {result.path.stat().st_size} bytes")


# ##################################################################
# real transparent image test — Nano Banana 2 + BiRefNet background removal


def test_generate_transparent_image():
    """Generate a transparent image via Nano Banana 2 + BiRefNet and verify RGBA PNG."""
    result = asyncio.run(
        generate_image(
            "A red fox standing on a white background",
            width=512,
            height=512,
            transparent=True,
            timeout=180.0,
        )
    )

    assert result.path.exists(), f"Image file not created: {result.path}"
    assert result.path.stat().st_size > 1000, "Image file suspiciously small"

    header = result.path.read_bytes()[:8]
    is_png = header[:4] == b"\x89PNG"
    is_jpeg = header[:2] == b"\xff\xd8"
    assert is_png or is_jpeg, f"Not a valid image, header: {header!r}"

    from PIL import Image
    with Image.open(result.path) as img:
        assert img.mode == "RGBA", f"Expected RGBA mode, got {img.mode}"

    print(f"\nGenerated transparent image at: {result.path}")
    print(f"File size: {result.path.stat().st_size} bytes")


# ##################################################################
# spark provider test — remote CUDA GPU


def _spark_reachable() -> bool:
    """Check if the spark image server is running."""
    import json
    from urllib.request import urlopen
    from urllib.error import URLError
    try:
        with urlopen("http://spark:8100/health", timeout=5) as resp:
            data = json.loads(resp.read())
            return data.get("status") == "ok"
    except (URLError, OSError):
        return False


def test_generate_spark_image():
    """Generate a 512x512 image via spark provider."""
    if not _spark_reachable():
        pytest.skip("Spark image server not reachable")

    result = asyncio.run(
        generate_image(
            "A green triangle on white background",
            width=512,
            height=512,
            provider="spark",
            timeout=300.0,
        )
    )

    assert result.path.exists()
    assert result.path.stat().st_size > 1000
    assert result.model_used.provider == "spark"
    assert result.model_used.model_id == "z-image-turbo"
    header = result.path.read_bytes()[:4]
    assert header == b"\x89PNG"


def test_generate_spark_transparent():
    """Generate a transparent image via spark (server-side BiRefNet)."""
    if not _spark_reachable():
        pytest.skip("Spark image server not reachable")

    result = asyncio.run(
        generate_image(
            "A red fox, white background",
            width=512,
            height=512,
            provider="spark",
            transparent=True,
            timeout=300.0,
        )
    )

    assert result.path.exists()
    from PIL import Image
    with Image.open(result.path) as img:
        assert img.mode == "RGBA"


def test_generate_default_provider_is_codex():
    """Verify codex is the default provider (native image_generation)."""
    import shutil
    if shutil.which("codex") is None:
        pytest.skip("codex CLI not on PATH")

    result = asyncio.run(
        generate_image(
            "A blue circle",
            width=512,
            height=512,
            timeout=600.0,
        )
    )

    assert result.model_used.provider == "codex"


# ##################################################################
# codex provider test — native image_generation via codex CLI


def _codex_available() -> bool:
    import shutil
    return shutil.which("codex") is not None


def test_generate_codex_image():
    """Generate an image via codex's native image_generation tool."""
    if not _codex_available():
        pytest.skip("codex CLI not on PATH")

    result = asyncio.run(
        generate_image(
            "A yellow star on a dark purple background",
            width=1024,
            height=1024,
            provider="codex",
            timeout=600.0,
        )
    )

    assert result.path.exists(), f"Image not created: {result.path}"
    assert result.path.stat().st_size > 1000
    assert result.model_used.provider == "codex"
    header = result.path.read_bytes()[:4]
    assert header == b"\x89PNG", f"Expected PNG, got header {header!r}"


def test_codex_rejects_input_image():
    """codex provider should reject image editing (input image) with INVALID_REQUEST."""
    if not _codex_available():
        pytest.skip("codex CLI not on PATH")

    import tempfile
    # write a tiny fake PNG header so the path-exists check passes
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
        tmp_path = tmp.name

    with pytest.raises(AgentError) as exc_info:
        asyncio.run(
            generate_image(
                "edit this",
                width=512,
                height=512,
                image=tmp_path,
                provider="codex",
                timeout=30.0,
            )
        )
    assert exc_info.value.kind == ErrorKind.INVALID_REQUEST


# ##################################################################
# mflux provider test — skip if not installed


def test_generate_mflux_image():
    """Generate an image via mflux provider (skipped if mflux not installed)."""
    try:
        import mflux  # noqa: F401  # pyright: ignore[reportMissingImports]
    except ImportError:
        pytest.skip("mflux not installed")

    result = asyncio.run(
        generate_image(
            "A blue circle on white",
            width=256,
            height=256,
            provider="mflux",
            timeout=120.0,
        )
    )

    assert result.path.exists()
    assert result.path.stat().st_size > 100
    assert result.model_used.provider == "mflux"


# ##################################################################
# error: missing API key


def test_missing_api_key_error(monkeypatch: pytest.MonkeyPatch):
    """Verify clear error when GEMINI_API_KEY is missing and Gemini is explicitly requested."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    # patch keyring.get_password to return None
    import keyring as _keyring_mod

    original_get = _keyring_mod.get_password
    monkeypatch.setattr(_keyring_mod, "get_password", lambda *a, **kw: None)  # noqa: ARG005

    try:
        with pytest.raises(AgentError) as exc_info:
            asyncio.run(
                generate_image(
                    "test",
                    width=512,
                    height=512,
                    provider="gemini",
                )
            )
        assert exc_info.value.kind == ErrorKind.AUTH
        assert "GEMINI_API_KEY" in str(exc_info.value)
    finally:
        monkeypatch.setattr(_keyring_mod, "get_password", original_get)


# ##################################################################
# error: missing transparent deps


def test_transparent_missing_deps_error(monkeypatch: pytest.MonkeyPatch):
    """Verify clear error when torch is missing for --transparent."""
    import builtins

    original_import = builtins.__import__

    def mock_import(name: str, *args, **kwargs):  # type: ignore
        if name == "torch":
            raise ImportError("No module named 'torch'")
        return original_import(name, *args, **kwargs)

    # we need to test the _ensure_transparent_deps function directly
    from daz_agent_sdk.capabilities.image import _ensure_transparent_deps

    monkeypatch.setattr(builtins, "__import__", mock_import)
    try:
        with pytest.raises(AgentError) as exc_info:
            _ensure_transparent_deps()
        assert exc_info.value.kind == ErrorKind.NOT_AVAILABLE
        assert "transparent" in str(exc_info.value)
    finally:
        monkeypatch.setattr(builtins, "__import__", original_import)


# ##################################################################
# anti-fabrication invariant — codex must NEVER return a stale image
#
# Regression for the worst-possible failure mode: codex's `exec` silently
# fails (e.g. "Reading additional input from stdin"), produces NO new image,
# yet the recovery logic used to hand back the newest image from a PREVIOUS
# generation and report success. Pretending to work is worse than failing.
# These tests drive _generate_codex with a faked subprocess and assert it
# RAISES when no fresh image appears, and only accepts an image written by
# THIS call.


class _FakeProc:
    def __init__(self, returncode: int, stdout: bytes, stderr: bytes, on_run=None):
        self.returncode = returncode
        self._stdout = stdout
        self._stderr = stderr
        self._on_run = on_run

    async def communicate(self):
        if self._on_run is not None:
            self._on_run()  # side effect: optionally create a "fresh" image
        return self._stdout, self._stderr

    def kill(self):
        pass


def _patch_codex(monkeypatch, tmp_path, thread_id, on_run):
    import shutil as _real_shutil

    from daz_agent_sdk.capabilities import image as image_mod

    monkeypatch.setattr(image_mod._shutil, "which", lambda _name: "/usr/bin/codex")
    monkeypatch.setattr(image_mod, "_CODEX_IMAGE_DIR", tmp_path)
    stdout = f'{{"type":"thread.started","thread_id":"{thread_id}"}}\n'.encode()

    async def _fake_exec(*_args, **_kwargs):
        return _FakeProc(0, stdout, b"", on_run=on_run)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)
    return _real_shutil


def test_codex_refuses_stale_image(monkeypatch, tmp_path):
    """codex exits 0 but creates NO new image; a stale image from a prior
    session exists. _generate_codex MUST raise, never return the stale one."""
    from daz_agent_sdk.capabilities.image import _generate_codex

    # a stale image from a previous generation, 100s old
    old_session = tmp_path / "old-session"
    old_session.mkdir()
    stale = old_session / "ig_0001.png"
    stale.write_bytes(b"\x89PNG\r\n\x1a\n" + b"STALE" * 64)
    import os as _os
    import time as _time
    old = _time.time() - 100
    _os.utime(stale, (old, old))

    out = tmp_path / "out.png"
    # codex "succeeds" (rc=0) but writes nothing new (on_run=None)
    _patch_codex(monkeypatch, tmp_path, thread_id="new-empty-session", on_run=None)

    with pytest.raises(AgentError) as exc:
        asyncio.run(_generate_codex(
            "a red apple", width=512, height=512, output_path=out, timeout=5.0,
        ))
    assert exc.value.kind == ErrorKind.INTERNAL
    assert not out.exists(), "must NOT have copied the stale image"


def test_codex_accepts_fresh_image(monkeypatch, tmp_path):
    """When codex DOES write a new image for this call, it is returned."""
    from daz_agent_sdk.capabilities.image import _generate_codex

    session = tmp_path / "this-session"

    def _make_fresh():
        session.mkdir(parents=True, exist_ok=True)
        (session / "ig_0001.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"FRESH" * 64)

    out = tmp_path / "out.png"
    _patch_codex(monkeypatch, tmp_path, thread_id="this-session", on_run=_make_fresh)

    asyncio.run(_generate_codex(
        "a red apple", width=512, height=512, output_path=out, timeout=5.0,
    ))
    assert out.exists() and out.read_bytes().startswith(b"\x89PNG")


# ##################################################################
# self-healing on codex safety refusal — diagnose + rewrite + retry
# (NEVER fabricate: if every rewrite still produces no image, raise)


def test_codex_self_heals_on_safety_refusal(monkeypatch, tmp_path):
    """First attempt refused → codex rewrites the prompt → retry succeeds."""
    from daz_agent_sdk.capabilities import image as image_mod

    out = tmp_path / "o.png"
    calls = {"img": 0, "rewrite": 0}
    prompts_seen = []

    async def fake_once(prompt, *, width, height, output_path, input_image=None, timeout=300.0):
        calls["img"] += 1
        prompts_seen.append(prompt)
        if calls["img"] == 1:
            return "blocked by the image safety system"  # refusal reason
        output_path.write_bytes(b"\x89PNG fresh image")  # success
        return None

    async def fake_rewrite(original, refusal, *, timeout):
        calls["rewrite"] += 1
        return "a safe rewritten prompt"

    monkeypatch.setattr(image_mod, "_codex_image_once", fake_once)
    monkeypatch.setattr(image_mod, "_codex_rewrite_safe_prompt", fake_rewrite)

    asyncio.run(image_mod._generate_codex(
        "risky prompt", width=512, height=512, output_path=out,
    ))
    assert out.exists()
    assert calls["img"] == 2 and calls["rewrite"] == 1
    assert prompts_seen == ["risky prompt", "a safe rewritten prompt"]


def test_codex_raises_when_all_rewrites_refused(monkeypatch, tmp_path):
    """Every attempt refused → raise (never fabricate), after MAX rewrites."""
    from daz_agent_sdk.capabilities import image as image_mod

    out = tmp_path / "o.png"
    calls = {"img": 0, "rewrite": 0}

    async def fake_once(prompt, *, width, height, output_path, input_image=None, timeout=300.0):
        calls["img"] += 1
        return "blocked by the image safety system"

    async def fake_rewrite(original, refusal, *, timeout):
        calls["rewrite"] += 1
        return f"rewrite {calls['rewrite']}"

    monkeypatch.setattr(image_mod, "_codex_image_once", fake_once)
    monkeypatch.setattr(image_mod, "_codex_rewrite_safe_prompt", fake_rewrite)

    with pytest.raises(AgentError) as exc:
        asyncio.run(image_mod._generate_codex(
            "risky", width=512, height=512, output_path=out,
        ))
    assert exc.value.kind == ErrorKind.INTERNAL
    assert not out.exists()
    assert calls["img"] == image_mod._MAX_SAFETY_REWRITES + 1
    assert calls["rewrite"] == image_mod._MAX_SAFETY_REWRITES
