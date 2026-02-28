from __future__ import annotations

import asyncio
import gc
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any
from uuid import UUID

from daz_agent_sdk.config import Config, load_config
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

    input_tensor = transform(rgb).unsqueeze(0).to(device)
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
# mflux generation path
async def _generate_mflux(
    prompt: str,
    *,
    width: int,
    height: int,
    output_path: Path,
    timeout: float = 120.0,
) -> None:
    try:
        import mflux  # noqa: F401
    except ImportError as exc:
        raise AgentError(
            "mflux not installed — pip install 'daz-agent-sdk[mflux]'",
            kind=ErrorKind.NOT_AVAILABLE,
        ) from exc

    def _call() -> None:
        from mflux import Flux1, Config as MfluxConfig
        flux = Flux1(
            model_alias="schnell",
            quantize=8,
        )
        image = flux.generate_image(
            seed=42,
            prompt=prompt,
            config=MfluxConfig(
                num_inference_steps=2,
                width=width,
                height=height,
            ),
        )
        image.save(path=str(output_path))

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
# generate image
# generates an image using Nano Banana 2 (default) or mflux.
# reads GEMINI_API_KEY from the environment (or keyring) for Nano Banana 2.
# saves the result to the output path (or a temp file).
async def generate_image(
    prompt: str,
    *,
    width: int,
    height: int,
    output: str | Path | None = None,
    tier: Tier = Tier.HIGH,
    transparent: bool = False,
    timeout: float = 120.0,
    provider: str | None = None,
    config: Config | None = None,
    logger: ConversationLogger | None = None,
    conversation_id: UUID | None = None,
) -> ImageResult:
    _ = config or load_config()  # ensure config is loaded

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

    if logger is not None:
        logger.log_event(
            "image_request",
            prompt=prompt,
            width=width,
            height=height,
            model=_NANO_BANANA_MODEL.model_id if provider != "mflux" else "mflux-schnell",
            tier=tier.value,
            transparent=transparent,
            provider=provider or "nano-banana-2",
        )

    # mflux provider path
    if provider == "mflux":
        await _generate_mflux(
            prompt,
            width=width,
            height=height,
            output_path=output_path,
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
                model_id="schnell",
                display_name="mflux schnell",
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

    # default: Nano Banana 2 via Gemini API
    aspect_ratio = _closest_aspect_ratio(width, height)
    image_size = _image_size(width, height)

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
        response = client.models.generate_content(
            model=_NANO_BANANA_MODEL.model_id,
            contents=[prompt],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(
                    aspect_ratio=aspect_ratio,
                    image_size=image_size,
                ),
            ),
        )
        for part in response.parts:
            if part.inline_data is not None:
                return part.inline_data.data
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
