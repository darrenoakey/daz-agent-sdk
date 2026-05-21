from __future__ import annotations

import asyncio
import base64
import fcntl
import gc
import os
import re
import shutil as _shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any
from uuid import UUID

from arbiter_client import ArbiterClient, ArbiterError as _ArbiterError, stage_file as _stage_file
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
    display_name="Spark z-image-turbo",
    capabilities=frozenset({Capability.IMAGE}),
    tier=Tier.HIGH,
    supports_streaming=False,
    supports_structured=False,
    supports_conversation=False,
)

# ##################################################################
# arbiter-based background removal on spark GPU
_ARBITER_URL = "http://10.0.0.254:8400"


async def _remove_background_spark(
    image_path: Path,
    *,
    timeout: float = 120.0,
    base_url: str | None = None,
) -> None:
    url = base_url or _ARBITER_URL

    def _call() -> None:
        client = ArbiterClient(base_url=url, timeout=30)
        image_b64 = base64.b64encode(image_path.read_bytes()).decode()
        try:
            s = client.run("background-remove", timeout=timeout, image=image_b64)
        except _ArbiterError as e:
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
# spark image generation via arbiter job API
# all spark image generation goes through arbiter at spark:8400.
# there is no direct spark:8100 endpoint anymore.


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
    model: str = "z-image-turbo",
) -> None:
    url = base_url or _ARBITER_URL

    def _call() -> None:
        client = ArbiterClient(base_url=url, timeout=30)
        params: dict[str, Any] = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "steps": steps,
        }
        job_type = "image-generate"
        if input_image is not None:
            job_type = "image-edit"
            # Stage via shared mount if configured, else base64
            try:
                params["image_file"] = _stage_file(input_image)
            except Exception:
                params["image"] = base64.b64encode(input_image.read_bytes()).decode()

        # Submit with model override in envelope (arbiter-specific field)
        try:
            job_id = client.submit_dict(job_type, params, model=model)
        except _ArbiterError as e:
            if "Connection" in str(e):
                raise AgentError(str(e), kind=ErrorKind.NOT_AVAILABLE) from e
            raise AgentError(str(e), kind=ErrorKind.INTERNAL) from e

        try:
            s = client.poll(job_id, interval=0.5, timeout=timeout)
        except _ArbiterError as e:
            raise AgentError(str(e), kind=ErrorKind.INTERNAL) from e

        result = s.get("result", {})
        image_b64 = result.get("image") or result.get("data")
        if not image_b64:
            raise AgentError(
                f"Arbiter {job_type} returned no image: {result}",
                kind=ErrorKind.INTERNAL,
            )
        output_path.write_bytes(base64.b64decode(image_b64))

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
# mflux model info (used by _generate_one)
_MFLUX_MODEL = ModelInfo(
    provider="mflux",
    model_id="z-image-turbo",
    display_name="mflux z-image-turbo",
    capabilities=frozenset({Capability.IMAGE}),
    tier=Tier.HIGH,
    supports_streaming=False,
    supports_structured=False,
    supports_conversation=False,
)


# ##################################################################
# codex native image generation (via OpenAI backend through `codex exec`)
# codex writes generated images to ~/.codex/generated_images/<session_id>/ig_*.png
_CODEX_MODEL = ModelInfo(
    provider="codex",
    model_id="codex-image-generation",
    display_name="Codex native image_generation",
    capabilities=frozenset({Capability.IMAGE}),
    tier=Tier.HIGH,
    supports_streaming=False,
    supports_structured=False,
    supports_conversation=False,
)

_CODEX_IMAGE_DIR = Path.home() / ".codex" / "generated_images"
_CODEX_SESSION_RE = re.compile(r"session id:\s*([0-9a-f-]+)", re.IGNORECASE)
# JSONL emits {"type":"thread.started","thread_id":"<uuid>"} as the very first event
_CODEX_THREAD_RE = re.compile(r'"thread_id"\s*:\s*"([0-9a-f-]+)"', re.IGNORECASE)


async def _generate_codex(
    prompt: str,
    *,
    width: int,
    height: int,
    output_path: Path,
    input_image: Path | list[Path] | None = None,
    timeout: float = 300.0,
) -> None:
    if _shutil.which("codex") is None:
        raise AgentError("codex CLI not found on PATH", kind=ErrorKind.NOT_AVAILABLE)

    input_images: list[Path] = []
    if input_image is not None:
        if isinstance(input_image, list):
            input_images = [Path(p) for p in input_image]
        else:
            input_images = [Path(input_image)]
        for ip in input_images:
            if not ip.exists():
                raise AgentError(
                    f"Input image not found: {ip}",
                    kind=ErrorKind.INVALID_REQUEST,
                )

    # prompt codex to call its native image_gen.imagegen tool only — no shell,
    # no generate_image skill, no external scripts. The tool schema only accepts
    # a `prompt` string, so size is mentioned in-prose. We recover the file from
    # the per-session ~/.codex/generated_images directory.
    aspect_hint = ""
    if width and height:
        if width > height:
            aspect_hint = " (landscape orientation)"
        elif height > width:
            aspect_hint = " (portrait orientation)"
        else:
            aspect_hint = " (square)"

    if input_images:
        action = (
            "EDIT the attached image" if len(input_images) == 1
            else f"COMBINE/EDIT the {len(input_images)} attached images into a single new image"
        )
        instruction = (
            f"You MUST call your built-in `image_gen.imagegen` tool exactly once to {action}. "
            "It is a NATIVE tool that you DO have access to in this environment — there is no need "
            "to apologise or claim you cannot generate images. Do NOT run any shell command. "
            "Do NOT call `~/bin/generate_image`, the `generate_image` skill, the `arbiter` tool, "
            "or any external script. The ONLY acceptable action is invoking `image_gen.imagegen` "
            "with a single `prompt` argument that fully describes the desired output image "
            f"(approx {width}x{height}{aspect_hint}). Use the attached image(s) as visual reference. "
            f"\n\nEdit description: {prompt}\n\n"
            "After the tool returns, reply with one short sentence confirming success."
        )
    else:
        instruction = (
            "You MUST call your built-in `image_gen.imagegen` tool exactly once. "
            "It is a NATIVE tool that you DO have access to in this environment — there is no "
            "need to apologise or claim you cannot generate images. Do NOT run any shell command. "
            "Do NOT call `~/bin/generate_image`, the `generate_image` skill, the `arbiter` tool, "
            "or any external script. The ONLY acceptable action is invoking `image_gen.imagegen` "
            "with a single `prompt` argument that fully describes the desired image "
            f"(approx {width}x{height}{aspect_hint}). "
            f"\n\nDesired image: {prompt}\n\n"
            "After the tool returns, reply with one short sentence confirming success."
        )

    # strip CLAUDECODE — it propagates from Claude Code sessions and causes
    # CLI tools like codex to segfault (exit -11). see ~/.claude/CLAUDE.md.
    child_env = os.environ.copy()
    child_env.pop("CLAUDECODE", None)

    cmd: list[str] = [
        "codex", "exec",
        "--dangerously-bypass-approvals-and-sandbox",
        "--skip-git-repo-check",
        "--ephemeral",
        "--json",
        "-m", "gpt-5.3-codex",
    ]
    for ip in input_images:
        cmd.extend(["-i", str(ip)])
    cmd.extend(["--", instruction])

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=child_env,
    )
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        proc.kill()
        raise AgentError(
            f"codex image generation timed out after {timeout}s",
            kind=ErrorKind.TIMEOUT,
        )

    stdout = stdout_bytes.decode(errors="replace")
    stderr = stderr_bytes.decode(errors="replace")

    if proc.returncode != 0:
        raise AgentError(
            f"codex exited with code {proc.returncode}: {stderr or stdout}",
            kind=ErrorKind.INTERNAL,
        )

    # locate the generated file — with --json, codex emits a thread.started
    # event with the session id as the first JSON line on stdout. Fall back to
    # the older "session id:" prose on stderr for compatibility.
    match = (
        _CODEX_THREAD_RE.search(stdout)
        or _CODEX_SESSION_RE.search(stderr)
        or _CODEX_SESSION_RE.search(stdout)
    )
    candidates: list[Path] = []
    if match:
        session_dir = _CODEX_IMAGE_DIR / match.group(1)
        if session_dir.is_dir():
            candidates = sorted(
                session_dir.glob("ig_*.png"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
    if not candidates and _CODEX_IMAGE_DIR.is_dir():
        # fallback: newest file anywhere under generated_images
        all_pngs = list(_CODEX_IMAGE_DIR.rglob("ig_*.png"))
        if all_pngs:
            candidates = sorted(all_pngs, key=lambda p: p.stat().st_mtime, reverse=True)

    if not candidates:
        raise AgentError(
            f"codex did not produce a generated image. stdout tail: {stdout[-500:]!r}",
            kind=ErrorKind.INTERNAL,
        )

    newest = candidates[0]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _shutil.copyfile(newest, output_path)


# ##################################################################
# generate with one provider
# dispatches to the correct backend and returns an ImageResult.
# raises AgentError on failure — caller handles fallback.
async def _generate_one(
    provider_name: str,
    prompt: str,
    *,
    width: int,
    height: int,
    output_path: Path,
    input_image: Path | list[Path] | None,
    tier: Tier,
    transparent: bool,
    timeout: float,
    cfg: Config,
    conv_id: UUID,
    model: str | None = None,
    steps_override: int | None = None,
) -> ImageResult:
    # most providers accept a single input image only; pick the first when
    # given a list. codex handles the full list natively.
    first_input: Path | None = None
    if isinstance(input_image, list):
        first_input = input_image[0] if input_image else None
    else:
        first_input = input_image
    # spark — CUDA-accelerated remote GPU
    if provider_name == "spark":
        steps = steps_override if steps_override and steps_override > 0 else get_image_steps(tier, cfg)
        spark_url = cfg.providers.get("arbiter", {}).get("base_url")
        effective_model = model or cfg.image.model or "z-image-turbo"
        await _generate_spark(
            prompt,
            width=width,
            height=height,
            steps=steps,
            output_path=output_path,
            input_image=first_input,
            transparent=transparent,
            timeout=timeout,
            base_url=spark_url,
            model=effective_model,
        )
        if transparent:
            arbiter_url = cfg.providers.get("arbiter", {}).get("base_url")
            await _remove_background_spark(output_path, timeout=timeout, base_url=arbiter_url)
        spark_model_info = ModelInfo(
            provider="spark",
            model_id=effective_model,
            display_name=f"Spark {effective_model}",
            capabilities=frozenset({Capability.IMAGE}),
            tier=Tier.HIGH,
            supports_streaming=False,
            supports_structured=False,
            supports_conversation=False,
        )
        return ImageResult(
            path=output_path,
            model_used=spark_model_info,
            conversation_id=conv_id,
            prompt=prompt,
            width=width,
            height=height,
        )

    # mflux — local Apple Silicon
    if provider_name == "mflux":
        steps = steps_override if steps_override and steps_override > 0 else get_image_steps(tier, cfg)
        await _generate_mflux(
            prompt,
            width=width,
            height=height,
            steps=steps,
            output_path=output_path,
            input_image=first_input,
            timeout=timeout,
        )
        if transparent:
            await _remove_background(output_path, timeout=timeout)
        return ImageResult(
            path=output_path,
            model_used=_MFLUX_MODEL,
            conversation_id=conv_id,
            prompt=prompt,
            width=width,
            height=height,
        )

    # codex — native image_generation tool via OpenAI backend
    if provider_name == "codex":
        await _generate_codex(
            prompt,
            width=width,
            height=height,
            output_path=output_path,
            input_image=input_image,
            timeout=timeout,
        )
        if transparent:
            await _remove_background(output_path, timeout=timeout)
        return ImageResult(
            path=output_path,
            model_used=_CODEX_MODEL,
            conversation_id=conv_id,
            prompt=prompt,
            width=width,
            height=height,
        )

    # nano-banana-2 — Gemini API
    if provider_name == "nano-banana-2":
        return await _generate_nano_banana(
            prompt,
            width=width,
            height=height,
            output_path=output_path,
            input_image=first_input,
            transparent=transparent,
            timeout=timeout,
            conv_id=conv_id,
        )

    raise AgentError(
        f"Unknown image provider: {provider_name}",
        kind=ErrorKind.INVALID_REQUEST,
    )


# ##################################################################
# nano banana 2 generation path
async def _generate_nano_banana(
    prompt: str,
    *,
    width: int,
    height: int,
    output_path: Path,
    input_image: Path | None,
    transparent: bool,
    timeout: float,
    conv_id: UUID,
) -> ImageResult:
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
        if input_image is not None:
            image_bytes = input_image.read_bytes()
            suffix = input_image.suffix.lower()
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

    return ImageResult(
        path=output_path,
        model_used=_NANO_BANANA_MODEL,
        conversation_id=conv_id,
        prompt=prompt,
        width=width,
        height=height,
    )


# ##################################################################
# generate image
# generates an image using the configured provider chain.
# tries the primary provider, then each fallback from config on failure.
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
    cfg = config or load_config()

    # validate input image(s) if provided. accept a single path or a list.
    # for the codex provider, multiple input images are forwarded as multiple
    # `-i` flags; other providers will use the first image only.
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

    # A model only exists on one provider — model without provider is an error.
    if model and not provider:
        raise AgentError(
            f"model={model!r} specified without provider — a model requires its provider",
            kind=ErrorKind.INVALID_REQUEST,
        )

    # When provider is explicit, use only that provider — no fallback chain.
    if provider:
        chain = [provider]
        fallbacks: list[str] = []
    else:
        primary = "codex"
        # default fallback order when the config doesn't specify one
        default_fallbacks = ["spark", "nano-banana-2", "mflux"]
        configured = list(cfg.image.fallback) if cfg.image.fallback else default_fallbacks
        fallbacks = [fb for fb in configured if fb != primary]
        chain = [primary] + fallbacks

    effective_primary = chain[0]
    if logger is not None:
        model_name = {
            "spark": _SPARK_MODEL.model_id,
            "mflux": "mflux-z-image-turbo",
            "codex": _CODEX_MODEL.model_id,
        }.get(effective_primary, _NANO_BANANA_MODEL.model_id)
        logger.log_event(
            "image_request",
            prompt=prompt,
            width=width,
            height=height,
            model=model_name,
            tier=tier.value,
            transparent=transparent,
            provider=effective_primary,
            fallbacks=fallbacks,
        )

    last_error: AgentError | None = None
    for provider_name in chain:
        try:
            if provider_name != effective_primary and logger is not None:
                logger.log_event("image_fallback", provider=provider_name, reason=str(last_error))
            result = await _generate_one(
                provider_name,
                prompt,
                width=width,
                height=height,
                output_path=output_path,
                input_image=input_image_path,
                tier=tier,
                transparent=transparent,
                timeout=timeout,
                cfg=cfg,
                conv_id=conv_id,
                model=model,
                steps_override=steps,
            )
            if logger is not None:
                logger.log_event("image_complete", path=str(output_path), provider=provider_name)
            return result
        except AgentError as err:
            last_error = err
            continue

    # all providers failed — raise the last error
    assert last_error is not None
    raise last_error
