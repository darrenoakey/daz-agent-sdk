from __future__ import annotations

import asyncio
import base64
import json
import socket
import tempfile
import uuid
from pathlib import Path
from typing import Any
from uuid import UUID

import aiohttp

from daz_agent_sdk.config import Config
from daz_agent_sdk.logging_ import ConversationLogger
from daz_agent_sdk.types import AgentError, Capability, ErrorKind, ImageResult, ModelInfo, Tier


# ##################################################################
# codex model info — image generation always goes through the mac mini
# image_generation_service (HTTP at :8830), which is backed by ChatGPT's
# image_generation tool over the codex responses endpoint.
_CODEX_MODEL = ModelInfo(
    provider="codex",
    model_id="macmini-image-service",
    display_name="Mac mini image generation service",
    capabilities=frozenset({Capability.IMAGE}),
    tier=Tier.HIGH,
    supports_streaming=False,
    supports_structured=False,
    supports_conversation=False,
)

_IMAGE_SERVICE_URL = "http://127.0.0.1:18831"
_LOCAL_IMAGE_SERVICE_URL = "http://127.0.0.1:8830"


def _image_service_url(hostname: str | None = None) -> str:
    """Use only loopback: native service on macmini, Auto-managed tunnel elsewhere."""
    machine = (hostname if hostname is not None else socket.gethostname()).split(".", 1)[0].lower()
    return _LOCAL_IMAGE_SERVICE_URL if machine == "macmini" else _IMAGE_SERVICE_URL


# ##################################################################
# arbiter-based background removal on the spark GPU.
#
# NOTE: this is a SEPARATE service from image generation. Image generation
# always goes through the mac mini image_generation_service at :8830 (codex).
# Background removal is a distinct arbiter job type ("background-remove") that
# runs BiRefNet on the spark GPU box. Agent.remove_background() is the only
# caller; it is NOT part of the image-generation provider chain.
_ARBITER_URL = "http://10.0.0.254:8400"


def _import_arbiter() -> Any:
    """Lazily import arbiter_client. Used only by background removal below."""
    try:
        import arbiter_client
    except ImportError as exc:
        raise AgentError(
            "background removal requires arbiter_client — pip install -e arbiter-client",
            kind=ErrorKind.NOT_AVAILABLE,
        ) from exc
    return arbiter_client


async def _remove_background_spark(
    image_path: Path,
    *,
    timeout: float = 120.0,
    base_url: str | None = None,
) -> None:
    url = base_url or _ARBITER_URL

    def _call() -> None:
        arbiter = _import_arbiter()
        client = arbiter.ArbiterClient(base_url=url, timeout=30)
        image_b64 = base64.b64encode(image_path.read_bytes()).decode()
        try:
            s = client.run("background-remove", timeout=timeout, image=image_b64)
        except arbiter.ArbiterError as e:
            if "Connection" in str(e) or "Unreachable" in str(e).lower():
                raise AgentError(str(e), kind=ErrorKind.NOT_AVAILABLE) from e
            raise AgentError(str(e), kind=ErrorKind.INTERNAL) from e
        result = s.get("result", {})
        image_b64_out = result.get("image") or result.get("data")
        if not image_b64_out:
            raise AgentError(
                f"Arbiter background-remove returned no image: {result}",
                kind=ErrorKind.INTERNAL,
            )
        image_path.write_bytes(base64.b64decode(image_b64_out))

    loop = asyncio.get_event_loop()
    try:
        await asyncio.wait_for(
            loop.run_in_executor(None, _call),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        raise AgentError(
            f"Arbiter background-remove timed out after {timeout}s",
            kind=ErrorKind.TIMEOUT,
        )


# ##################################################################
# image service helpers


def _service_input_images(input_image: Path | list[Path] | None) -> list[str]:
    """Encode input/reference images as base64 strings for the service."""
    if input_image is None:
        return []
    paths = input_image if isinstance(input_image, list) else [input_image]
    encoded: list[str] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            raise AgentError(
                f"Input image not found: {path}",
                kind=ErrorKind.INVALID_REQUEST,
            )
        encoded.append(base64.b64encode(path.read_bytes()).decode("ascii"))
    return encoded


async def _service_json(
    session: aiohttp.ClientSession,
    method: str,
    path: str,
    *,
    payload: dict[str, Any] | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    async with session.request(method, (base_url or _image_service_url()) + path, json=payload) as response:
        text = await response.text()
        if response.status >= 400:
            raise AgentError(
                f"image service {method} {path} returned HTTP {response.status}: {text}",
                kind=ErrorKind.INTERNAL,
            )
        try:
            data = json.loads(text)
        except ValueError as exc:
            raise AgentError(
                f"image service {method} {path} returned invalid JSON: {text[:200]}",
                kind=ErrorKind.INTERNAL,
            ) from exc
        if not isinstance(data, dict):
            raise AgentError(
                f"image service {method} {path} returned non-object JSON",
                kind=ErrorKind.INTERNAL,
            )
        return data


async def _service_image(session: aiohttp.ClientSession, path: str, *, base_url: str | None = None) -> bytes:
    async with session.get((base_url or _image_service_url()) + path) as response:
        data = await response.read()
        if response.status >= 400:
            raise AgentError(
                f"image service GET {path} returned HTTP {response.status}: {data[:200]!r}",
                kind=ErrorKind.INTERNAL,
            )
        if not data.startswith(b"\x89PNG\r\n\x1a\n"):
            content_type = response.headers.get("content-type", "")
            raise AgentError(
                f"image service returned non-PNG data ({content_type})",
                kind=ErrorKind.INTERNAL,
            )
        return data


def _write_service_image(image_data: bytes, output_path: Path, transparent: bool) -> None:
    """Write the PNG bytes from the service to output_path.

    The service always returns PNG. If the caller asked for JPEG, convert via
    Pillow. Transparent output must stay PNG.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        if transparent:
            raise AgentError(
                "transparent image output must be PNG, not JPEG",
                kind=ErrorKind.INVALID_REQUEST,
            )
        try:
            from PIL import Image
        except ImportError as exc:
            raise AgentError(
                "Pillow is required to save image service PNG output as JPEG",
                kind=ErrorKind.NOT_AVAILABLE,
            ) from exc
        from io import BytesIO

        with Image.open(BytesIO(image_data)) as image:
            image.convert("RGB").save(output_path, "JPEG", quality=92)
        return
    output_path.write_bytes(image_data)


# ##################################################################
# generate one image via the mac mini image_generation_service.
async def _generate_igs(
    prompt: str,
    *,
    width: int,
    height: int,
    output_path: Path,
    input_image: Path | list[Path] | None = None,
    transparent: bool = False,
    timeout: float = 300.0,
) -> None:
    payload: dict[str, Any] = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "transparent": transparent,
    }
    sources = _service_input_images(input_image)
    if sources:
        payload["source_images"] = sources

    deadline = asyncio.get_running_loop().time() + timeout
    client_timeout = aiohttp.ClientTimeout(total=60)
    service_url = _image_service_url()
    async with aiohttp.ClientSession(timeout=client_timeout) as session:
        created = await _service_json(session, "POST", "/jobs", payload=payload, base_url=service_url)
        job_id = str(created.get("id", "")).strip()
        if not job_id:
            raise AgentError(
                f"image service returned no job id: {created}",
                kind=ErrorKind.INTERNAL,
            )

        while True:
            status = await _service_json(session, "GET", f"/jobs/{job_id}", base_url=service_url)
            state = status.get("status")
            if state == "done":
                break
            if state == "failed":
                raise AgentError(
                    f"image service job {job_id} failed after {status.get('attempts', 0)} attempts: "
                    f"{status.get('error', '')}",
                    kind=ErrorKind.INTERNAL,
                )
            if state not in {"queued", "running"}:
                raise AgentError(
                    f"image service job {job_id} returned unknown status {state!r}",
                    kind=ErrorKind.INTERNAL,
                )
            if asyncio.get_running_loop().time() >= deadline:
                raise AgentError(
                    f"image service job {job_id} timed out after {timeout:.0f}s",
                    kind=ErrorKind.TIMEOUT,
                )
            await asyncio.sleep(2)

        image_data = await _service_image(session, f"/jobs/{job_id}/image", base_url=service_url)
    _write_service_image(image_data, output_path, transparent)


# ##################################################################
# generate image
#
# There is exactly one image generation path: the mac mini
# image_generation_service at http://10.0.0.46:8830, which is backed by
# ChatGPT's image_generation tool over the codex responses endpoint. The
# legacy spark (flux), mflux, nano-banana-2, and codex-CLI backends have
# been removed — codex is the only provider.
async def generate_image(
    prompt: str,
    *,
    width: int,
    height: int,
    output: str | Path | None = None,
    image: str | Path | list[str | Path] | None = None,
    tier: Tier = Tier.HIGH,
    transparent: bool = False,
    timeout: float = 120.0,
    provider: str | None = None,
    model: str | None = None,
    steps: int | None = None,
    config: Config | None = None,
    logger: ConversationLogger | None = None,
    conversation_id: UUID | None = None,
) -> ImageResult:
    # validate input image(s) if provided. accept a single path or a list.
    input_image_path: Path | list[Path] | None = None
    if image is not None:
        if isinstance(image, list):
            paths = [Path(p) for p in image]
            for p in paths:
                if not p.exists():
                    raise AgentError(
                        f"Input image not found: {p}",
                        kind=ErrorKind.INVALID_REQUEST,
                    )
            input_image_path = paths if len(paths) > 1 else (paths[0] if paths else None)
        else:
            single = Path(image)
            if not single.exists():
                raise AgentError(
                    f"Input image not found: {single}",
                    kind=ErrorKind.INVALID_REQUEST,
                )
            input_image_path = single

    if output is None:
        tmp = tempfile.NamedTemporaryFile(
            suffix=".png",
            prefix="agent_sdk_img_",
            delete=False,
        )
        tmp.close()
        output_path = Path(tmp.name)
    else:
        output_path = Path(output)

    conv_id = conversation_id or uuid.uuid4()

    # codex is the only supported provider. Reject explicit requests for any
    # other provider loudly rather than silently re-routing.
    if provider is not None and provider != "codex":
        raise AgentError(
            f"image provider {provider!r} is not supported — only 'codex' "
            f"(mac mini image_generation_service) is available.",
            kind=ErrorKind.INVALID_REQUEST,
        )

    # `model` is accepted for backward compatibility with old callers but has
    # no effect: the mac mini service owns model selection (gpt-5.5 via codex).
    # `steps` is likewise a no-op now (the service decides).

    if logger is not None:
        logger.log_event(
            "image_request",
            prompt=prompt,
            width=width,
            height=height,
            model=_CODEX_MODEL.model_id,
            tier=tier.value,
            transparent=transparent,
            provider="codex",
            fallbacks=[],
        )

    try:
        await _generate_igs(
            prompt,
            width=width,
            height=height,
            output_path=output_path,
            input_image=input_image_path,
            transparent=transparent,
            timeout=timeout,
        )
    except AgentError:
        raise
    except Exception as err:
        raise AgentError(
            f"image generation failed: {err}",
            kind=ErrorKind.INTERNAL,
        ) from err

    if logger is not None:
        logger.log_event("image_complete", path=str(output_path), provider="codex")

    return ImageResult(
        path=output_path,
        model_used=_CODEX_MODEL,
        conversation_id=conv_id,
        prompt=prompt,
        width=width,
        height=height,
    )
