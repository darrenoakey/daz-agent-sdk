from __future__ import annotations

import asyncio
import fcntl
import gc
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any
from uuid import UUID

from daz_agent_sdk.config import Config, get_image_steps, load_config
from daz_agent_sdk.logging_ import ConversationLogger
from daz_agent_sdk.types import AgentError, Capability, ErrorKind, ImageResult, ModelInfo, Tier


# ##################################################################
# nano banana 2 model info
_NANO_BANANA_MODEL = ModelInfo(
    provider="gemini",
    model_id="gemini-3.1-flash-image-preview",
    display_name="Nano Banana 2",
    capabilities=frozenset({Capability.IMAGE}),
    tier=Tier.HIGH,
    supports_streaming=False,
    supports_structured=False,
    supports_conversation=False,
)

# ##################################################################
# aspect ratio mapping
# nano banana 2 supports specific aspect ratios; pick the closest one
_ASPECT_RATIOS = ["1:1", "3:4", "4:3", "9:16", "16:9"]


def _closest_aspect_ratio(width: int, height: int) -> str:
    target = width / height
    best = _ASPECT_RATIOS[0]
    best_diff = float("inf")
    for ar in _ASPECT_RATIOS:
        w, h = ar.split(":")
        ratio = int(w) / int(h)
        diff = abs(ratio - target)
        if diff < best_diff:
            best_diff = diff
            best = ar
    return best


# ##################################################################
# image size mapping
# nano banana 2 supports 0.5K, 1K, 2K, 4K — pick based on max dimension
def _image_size(width: int, height: int) -> str:
    max_dim = max(width, height)
    if max_dim <= 512:
        return "0.5K"
    if max_dim <= 1024:
        return "1K"
    if max_dim <= 2048:
        return "2K"
    return "4K"


# ##################################################################
# BiRefNet background removal — inline, no subprocess
# cached model singleton (like EKEventStore pattern)
_birefnet_model: Any = None
_birefnet_device: Any = None
_BIREFNET_MODEL_NAME = "ZhengPeng7/BiRefNet_HR"
_BIREFNET_INPUT_SIZE = (1024, 1024)
_BIREFNET_CACHE_DIR = Path.home() / ".cache" / "background-removal" / "huggingface"


def _ensure_transparent_deps() -> None:
    try:
        import torch  # noqa: F401
        from PIL import Image  # noqa: F401
        from torchvision import transforms  # noqa: F401
        from transformers import AutoModelForImageSegmentation  # noqa: F401
    except ImportError as exc:
        raise AgentError(
            "Background removal requires: pip install 'daz-agent-sdk[transparent]'",
            kind=ErrorKind.NOT_AVAILABLE,
        ) from exc


def _get_birefnet_model() -> tuple[Any, Any]:
    global _birefnet_model, _birefnet_device
    if _birefnet_model is not None:
        return _birefnet_model, _birefnet_device

    import torch
    from transformers import AutoModelForImageSegmentation
    cache_dir = _BIREFNET_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    for env_var in ("HF_HOME", "TRANSFORMERS_CACHE", "TORCH_HOME", "XDG_CACHE_HOME"):
        if not os.environ.get(env_var):
            os.environ[env_var] = str(cache_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForImageSegmentation.from_pretrained(
        _BIREFNET_MODEL_NAME,
        trust_remote_code=True,
        cache_dir=str(cache_dir),
    )
    model.eval()
    model = model.to(device).float()

    _birefnet_model = model
    _birefnet_device = device
    return model, device


def _remove_background_sync(image_path: Path) -> None:
    import torch
    from PIL import Image, ImageOps
    from torchvision import transforms

    model, device = _get_birefnet_model()

    transform = transforms.Compose([
        transforms.Resize(_BIREFNET_INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    with Image.open(image_path) as img:
        original = img.copy()
    original = ImageOps.exif_transpose(original)
    rgb = original.convert("RGB") if original.mode != "RGB" else original

    input_tensor = torch.Tensor(transform(rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        prediction = outputs[-1].sigmoid().cpu()

    alpha_tensor = prediction[0].squeeze()
    alpha_image = transforms.ToPILImage()(alpha_tensor)
    alpha_mask = alpha_image.resize(original.size)

    result = original.convert("RGBA")
    result.putalpha(alpha_mask)
    result.save(image_path, format="PNG")

    gc.collect()


async def _remove_background(image_path: Path, *, timeout: float = 120.0) -> None:
    _ensure_transparent_deps()
    loop = asyncio.get_event_loop()
    try:
        await asyncio.wait_for(
            loop.run_in_executor(None, _remove_background_sync, image_path),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        raise AgentError(
            f"Background removal timed out after {timeout}s",
            kind=ErrorKind.TIMEOUT,
        )


# ##################################################################
# system-wide lock for mflux — only one generation at a time across all processes
_MFLUX_LOCK_PATH = Path(tempfile.gettempdir()) / "daz-agent-sdk-mflux.lock"


# ##################################################################
# mflux generation path
async def _generate_mflux(
    prompt: str,
    *,
    width: int,
    height: int,
    steps: int,
    output_path: Path,
    input_image: Path | None = None,
    timeout: float = 120.0,
) -> None:
    try:
        import mflux  # noqa: F401  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        raise AgentError(
            "mflux not installed — pip install 'daz-agent-sdk[mflux]'",
            kind=ErrorKind.NOT_AVAILABLE,
        ) from exc

    def _call() -> None:
        from mflux.models.z_image import ZImageTurbo  # pyright: ignore[reportMissingImports]

        # system-wide exclusive lock — blocks until no other process is generating
        with open(_MFLUX_LOCK_PATH, "w") as lock_fd:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            try:
                model = ZImageTurbo(quantize=8)
                gen_kwargs: dict[str, Any] = {
                    "seed": 42,
                    "prompt": prompt,
                    "num_inference_steps": steps,
                    "width": width,
                    "height": height,
                }
                if input_image is not None:
                    gen_kwargs["image_path"] = str(input_image)
                image = model.generate_image(**gen_kwargs)
                image.save(path=str(output_path), overwrite=True)  # pyright: ignore[reportCallIssue]
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)

    loop = asyncio.get_event_loop()
    try:
        await asyncio.wait_for(
            loop.run_in_executor(None, _call),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        raise AgentError(
            f"mflux generation timed out after {timeout}s",
            kind=ErrorKind.TIMEOUT,
        )


# ##################################################################
# spark model info
_SPARK_MODEL = ModelInfo(
    provider="spark",
    model_id="z-image-turbo",
    display_name="Spark Z-Image Turbo (FLUX.1-schnell)",
    capabilities=frozenset({Capability.IMAGE}),
    tier=Tier.HIGH,
    supports_streaming=False,
    supports_structured=False,
    supports_conversation=False,
)

# ##################################################################
# arbiter-based background removal on spark GPU
_ARBITER_DEFAULT_URL = "http://spark:8400"


async def _remove_background_spark(
    image_path: Path,
    *,
    timeout: float = 120.0,
    base_url: str | None = None,
) -> None:
    import base64
    import json
    import time
    from urllib.request import Request, urlopen
    from urllib.error import URLError

    url = base_url or os.environ.get("ARBITER_URL") or _ARBITER_DEFAULT_URL

    def _call() -> None:
        # submit background-remove job
        image_b64 = base64.b64encode(image_path.read_bytes()).decode()
        payload = json.dumps({
            "type": "background-remove",
            "params": {"image": image_b64},
        }).encode()
        req = Request(
            f"{url}/v1/jobs",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
        except URLError as e:
            raise AgentError(
                f"Arbiter unreachable at {url}: {e}",
                kind=ErrorKind.NOT_AVAILABLE,
            ) from e

        job_id = data.get("job_id")
        if not job_id:
            raise AgentError(
                f"Arbiter returned no job_id: {data}",
                kind=ErrorKind.INTERNAL,
            )

        # poll until done
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            poll_req = Request(f"{url}/v1/jobs/{job_id}", method="GET")
            with urlopen(poll_req, timeout=10) as resp:
                status_data = json.loads(resp.read())
            status = status_data.get("status")
            if status == "completed":
                result = status_data.get("result", {})
                image_b64_out = result.get("image") or result.get("data")
                if not image_b64_out:
                    raise AgentError(
                        f"Arbiter background-remove returned no image: {status_data}",
                        kind=ErrorKind.INTERNAL,
                    )
                image_path.write_bytes(base64.b64decode(image_b64_out))
                return
            if status == "failed":
                raise AgentError(
                    f"Arbiter background-remove failed: {status_data.get('error')}",
                    kind=ErrorKind.INTERNAL,
                )
            time.sleep(1)

        raise AgentError(
            f"Arbiter background-remove timed out after {timeout}s",
            kind=ErrorKind.TIMEOUT,
        )

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
# spark image server — CUDA-accelerated FLUX.1-schnell on remote GPU
_SPARK_DEFAULT_URL = "http://spark:8100"


async def _generate_spark(
    prompt: str,
    *,
    width: int,
    height: int,
    steps: int,
    output_path: Path,
    input_image: Path | None = None,
    transparent: bool = False,
    timeout: float = 300.0,
    base_url: str | None = None,
) -> None:
    import base64
    import json
    from urllib.request import Request, urlopen
    from urllib.error import URLError

    url = base_url or os.environ.get("SPARK_IMAGE_URL") or _SPARK_DEFAULT_URL

    def _call() -> None:
        if input_image is not None:
            # img2img via multipart form
            import mimetypes
            boundary = "----AgentSDKBoundary"
            body = b""
            fields = {
                "prompt": prompt,
                "model": "z-image-turbo",
                "width": str(width),
                "height": str(height),
                "steps": str(steps),
                "transparent": str(transparent).lower(),
            }
            for k, v in fields.items():
                body += f"--{boundary}\r\n".encode()
                body += f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode()
                body += f"{v}\r\n".encode()
            # image file
            mime = mimetypes.guess_type(str(input_image))[0] or "image/png"
            body += f"--{boundary}\r\n".encode()
            body += f'Content-Disposition: form-data; name="image"; filename="{input_image.name}"\r\n'.encode()
            body += f"Content-Type: {mime}\r\n\r\n".encode()
            body += input_image.read_bytes()
            body += f"\r\n--{boundary}--\r\n".encode()
            req = Request(
                f"{url}/v1/images/edit",
                data=body,
                headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
                method="POST",
            )
        else:
            # text-to-image via JSON
            payload = json.dumps({
                "prompt": prompt,
                "model": "z-image-turbo",
                "width": width,
                "height": height,
                "steps": steps,
                "transparent": transparent,
            }).encode()
            req = Request(
                f"{url}/v1/images/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

        try:
            with urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read())
        except URLError as e:
            raise AgentError(
                f"Spark image server unreachable at {url}: {e}",
                kind=ErrorKind.NOT_AVAILABLE,
            ) from e

        if "image" not in data:
            raise AgentError(
                f"Spark returned no image data: {data}",
                kind=ErrorKind.INTERNAL,
            )
        output_path.write_bytes(base64.b64decode(data["image"]))

    loop = asyncio.get_event_loop()
    try:
        await asyncio.wait_for(
            loop.run_in_executor(None, _call),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        raise AgentError(
            f"Spark image generation timed out after {timeout}s",
            kind=ErrorKind.TIMEOUT,
        )


# ##################################################################
# generate image
# generates an image using spark (default), mflux, or Nano Banana 2.
# saves the result to the output path (or a temp file).
async def generate_image(
    prompt: str,
    *,
    width: int,
    height: int,
    output: str | Path | None = None,
    image: str | Path | None = None,
    tier: Tier = Tier.HIGH,
    transparent: bool = False,
    timeout: float = 120.0,
    provider: str | None = None,
    config: Config | None = None,
    logger: ConversationLogger | None = None,
    conversation_id: UUID | None = None,
) -> ImageResult:
    cfg = config or load_config()

    # validate input image if provided
    input_image_path: Path | None = None
    if image is not None:
        input_image_path = Path(image)
        if not input_image_path.exists():
            raise AgentError(
                f"Input image not found: {input_image_path}",
                kind=ErrorKind.INVALID_REQUEST,
            )

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

    effective_provider = provider or "spark"

    if logger is not None:
        model_name = {
            "spark": _SPARK_MODEL.model_id,
            "mflux": "mflux-z-image-turbo",
        }.get(effective_provider, _NANO_BANANA_MODEL.model_id)
        logger.log_event(
            "image_request",
            prompt=prompt,
            width=width,
            height=height,
            model=model_name,
            tier=tier.value,
            transparent=transparent,
            provider=effective_provider,
        )

    # spark provider path — CUDA-accelerated remote GPU
    if effective_provider == "spark":
        steps = get_image_steps(tier, cfg)
        spark_url = cfg.providers.get("spark", {}).get("base_url") if cfg else None
        await _generate_spark(
            prompt,
            width=width,
            height=height,
            steps=steps,
            output_path=output_path,
            input_image=input_image_path,
            transparent=transparent,
            timeout=timeout,
            base_url=spark_url,
        )
        if transparent:
            arbiter_url = cfg.providers.get("arbiter", {}).get("base_url") if cfg else None
            await _remove_background_spark(output_path, timeout=timeout, base_url=arbiter_url)
        if logger is not None:
            logger.log_event("image_complete", path=str(output_path))
        return ImageResult(
            path=output_path,
            model_used=_SPARK_MODEL,
            conversation_id=conv_id,
            prompt=prompt,
            width=width,
            height=height,
        )

    # mflux provider path (Apple Silicon local)
    if effective_provider == "mflux":
        steps = get_image_steps(tier, cfg)
        await _generate_mflux(
            prompt,
            width=width,
            height=height,
            steps=steps,
            output_path=output_path,
            input_image=input_image_path,
            timeout=timeout,
        )
        if transparent:
            await _remove_background(output_path, timeout=timeout)
        if logger is not None:
            logger.log_event("image_complete", path=str(output_path))
        return ImageResult(
            path=output_path,
            model_used=ModelInfo(
                provider="mflux",
                model_id="z-image-turbo",
                display_name="mflux z-image-turbo",
                capabilities=frozenset({Capability.IMAGE}),
                tier=Tier.HIGH,
                supports_streaming=False,
                supports_structured=False,
                supports_conversation=False,
            ),
            conversation_id=conv_id,
            prompt=prompt,
            width=width,
            height=height,
        )

    # Nano Banana 2 via Gemini API (opt-in with provider="nano-banana-2")
    aspect_ratio = _closest_aspect_ratio(width, height)

    try:
        from google import genai  # type: ignore[attr-defined]
        from google.genai import types  # type: ignore[attr-defined]
    except ImportError as exc:
        raise AgentError(
            "google-genai package not installed — pip install google-genai",
            kind=ErrorKind.NOT_AVAILABLE,
        ) from exc

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        try:
            import keyring
            api_key = keyring.get_password("google", "api_key")
        except Exception:
            pass
    if not api_key:
        raise AgentError(
            "GEMINI_API_KEY not set and not found in keyring",
            kind=ErrorKind.AUTH,
        )

    client = genai.Client(api_key=api_key)

    def _call() -> bytes:
        contents: list[Any] = []
        if input_image_path is not None:
            image_bytes = input_image_path.read_bytes()
            suffix = input_image_path.suffix.lower()
            mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".gif": "image/gif"}
            mime_type = mime_map.get(suffix, "image/png")
            contents.append(types.Part(inline_data=types.Blob(data=image_bytes, mime_type=mime_type)))
        contents.append(prompt)
        response = client.models.generate_content(
            model=_NANO_BANANA_MODEL.model_id,
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(
                    aspect_ratio=aspect_ratio,
                ),
            ),
        )
        for part in response.parts or []:  # type: ignore[union-attr]
            if part.inline_data is not None:
                data = part.inline_data.data
                if data is not None:
                    return data
        raise AgentError(
            "Nano Banana 2 returned no image data",
            kind=ErrorKind.INTERNAL,
        )

    try:
        loop = asyncio.get_event_loop()
        image_bytes = await asyncio.wait_for(
            loop.run_in_executor(None, _call),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        raise AgentError(
            f"Image generation timed out after {timeout}s",
            kind=ErrorKind.TIMEOUT,
        )
    except AgentError:
        raise
    except Exception as err:
        msg = str(err).lower()
        if "429" in msg or "quota" in msg or "rate" in msg:
            kind = ErrorKind.RATE_LIMIT
        elif "401" in msg or "403" in msg or "api_key" in msg:
            kind = ErrorKind.AUTH
        else:
            kind = ErrorKind.INTERNAL
        raise AgentError(str(err), kind=kind) from err

    output_path.write_bytes(image_bytes)

    if transparent:
        await _remove_background(output_path, timeout=timeout)

    if logger is not None:
        logger.log_event("image_complete", path=str(output_path))

    return ImageResult(
        path=output_path,
        model_used=_NANO_BANANA_MODEL,
        conversation_id=conv_id,
        prompt=prompt,
        width=width,
        height=height,
    )
