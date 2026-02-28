from __future__ import annotations

import asyncio
import tempfile
import uuid
from pathlib import Path
from uuid import UUID

from daz_agent_sdk.config import Config, get_image_steps, load_config
from daz_agent_sdk.logging_ import ConversationLogger
from daz_agent_sdk.types import AgentError, Capability, ErrorKind, ImageResult, ModelInfo, Tier


# ##################################################################
# local image model info
# placeholder ModelInfo for the local generate_image subprocess tool.
# used as model_used in ImageResult when no provider model is involved.
_LOCAL_IMAGE_MODEL = ModelInfo(
    provider="local",
    model_id="generate_image",
    display_name="Local Image Generator",
    capabilities=frozenset({Capability.IMAGE}),
    tier=Tier.HIGH,
    supports_streaming=False,
    supports_structured=False,
    supports_conversation=False,
)


# ##################################################################
# resolve steps
# determines the inference step count from tier using config,
# unless the caller has supplied an explicit override.
def _resolve_steps(tier: Tier, steps: int | None, config: Config | None) -> int:
    if steps is not None:
        return steps
    return get_image_steps(tier, config)


# ##################################################################
# build command
# constructs the generate_image subprocess argument list from the
# supplied parameters. when transparent=True, passes --transparent
# which enforces .png output and runs background removal automatically.
def _build_command(
    prompt: str,
    *,
    width: int,
    height: int,
    output: Path,
    model: str,
    steps: int,
    transparent: bool = False,
    image: Path | None = None,
    image_strength: float | None = None,
    guidance: float | None = None,
    quantize: int | None = None,
    seed: int | None = None,
) -> list[str]:
    cmd = [
        "generate_image",
        "--prompt", prompt,
        "--width", str(width),
        "--height", str(height),
        "--output", str(output),
        "--model", model,
        "--steps", str(steps),
    ]
    if transparent:
        cmd.append("--transparent")
    if image is not None:
        cmd.extend(["--image", str(image)])
    if image_strength is not None:
        cmd.extend(["--image-strength", str(image_strength)])
    if guidance is not None:
        cmd.extend(["--guidance", str(guidance)])
    if quantize is not None:
        cmd.extend(["--quantize", str(quantize)])
    if seed is not None:
        cmd.extend(["--seed", str(seed)])
    return cmd




# ##################################################################
# run subprocess
# runs a subprocess using asyncio.create_subprocess_exec and waits
# for it to complete. raises AgentError if the process exits non-zero.
async def _run_subprocess(args: list[str], *, timeout: float, label: str) -> None:
    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            proc.kill()
            await proc.communicate()
            raise AgentError(
                f"{label} timed out after {timeout}s",
                kind=ErrorKind.TIMEOUT,
            ) from exc

        if proc.returncode != 0:
            stderr_text = stderr.decode(errors="replace").strip()
            raise AgentError(
                f"{label} failed (exit {proc.returncode}): {stderr_text}",
                kind=ErrorKind.INTERNAL,
            )
    except AgentError:
        raise
    except OSError as exc:
        raise AgentError(
            f"{label} could not be started: {exc}",
            kind=ErrorKind.NOT_AVAILABLE,
        ) from exc


# ##################################################################
# generate image
# generates an image from a text prompt using the local generate_image
# subprocess tool. resolves inference steps from the tier config unless
# steps is provided explicitly. if transparent=True, runs
# remove-background on the output after generation.
async def generate_image(
    prompt: str,
    *,
    width: int,
    height: int,
    output: str | Path | None = None,
    tier: Tier = Tier.HIGH,
    steps: int | None = None,
    transparent: bool = False,
    model: str | None = None,
    image: str | Path | None = None,
    image_strength: float | None = None,
    guidance: float | None = None,
    quantize: int | None = None,
    seed: int | None = None,
    timeout: float = 600.0,
    config: Config | None = None,
    logger: ConversationLogger | None = None,
    conversation_id: UUID | None = None,
) -> ImageResult:
    cfg = config or load_config()
    resolved_steps = _resolve_steps(tier, steps, cfg)
    resolved_model = model or cfg.image.model

    if output is None:
        suffix = ".png" if transparent else ".jpg"
        tmp = tempfile.NamedTemporaryFile(
            suffix=suffix,
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
            model=resolved_model,
            steps=resolved_steps,
            tier=tier.value,
            transparent=transparent,
        )

    cmd = _build_command(
        prompt,
        width=width,
        height=height,
        output=output_path,
        model=resolved_model,
        steps=resolved_steps,
        transparent=transparent,
        image=Path(image) if image is not None else None,
        image_strength=image_strength,
        guidance=guidance,
        quantize=quantize,
        seed=seed,
    )

    await _run_subprocess(cmd, timeout=timeout, label="generate_image")

    if logger is not None:
        logger.log_event("image_complete", path=str(output_path))

    return ImageResult(
        path=output_path,
        model_used=_LOCAL_IMAGE_MODEL,
        conversation_id=conv_id,
        prompt=prompt,
        width=width,
        height=height,
    )
