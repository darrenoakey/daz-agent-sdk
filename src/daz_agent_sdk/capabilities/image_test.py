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
    assert result.model_used.model_id == "gemini-3.1-flash-image-preview"

    header = result.path.read_bytes()[:8]
    assert header[:4] == b"\x89PNG", f"Not a valid PNG, header: {header!r}"
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
    assert header[:4] == b"\x89PNG", f"Not a valid PNG, header: {header!r}"

    from PIL import Image
    with Image.open(result.path) as img:
        assert img.mode == "RGBA", f"Expected RGBA mode, got {img.mode}"

    print(f"\nGenerated transparent image at: {result.path}")
    print(f"File size: {result.path.stat().st_size} bytes")


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
    """Verify clear error when GEMINI_API_KEY is missing."""
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
