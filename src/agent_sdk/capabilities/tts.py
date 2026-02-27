from __future__ import annotations

import asyncio
import tempfile
import uuid
from pathlib import Path
from uuid import UUID

from agent_sdk.logging_ import ConversationLogger
from agent_sdk.types import AgentError, AudioResult, Capability, ErrorKind, ModelInfo, Tier


# ##################################################################
# local tts model info
# placeholder ModelInfo for the local tts subprocess tool.
_LOCAL_TTS_MODEL = ModelInfo(
    provider="local",
    model_id="tts",
    display_name="Local TTS",
    capabilities=frozenset({Capability.TTS}),
    tier=Tier.HIGH,
    supports_streaming=False,
    supports_structured=False,
    supports_conversation=False,
)

# ##################################################################
# default voice
# used when the caller does not specify a voice
_DEFAULT_VOICE = "aiden"

# ##################################################################
# default output suffix
# audio format produced by the local tts tool
_DEFAULT_SUFFIX = ".wav"


# ##################################################################
# build command
# constructs the tts subprocess argument list from the supplied
# parameters. output path must already be resolved by the caller.
def _build_tts_command(
    text: str,
    *,
    voice: str,
    output: Path,
    speed: float,
) -> list[str]:
    return [
        "tts",
        "tts",
        "--text", text,
        "--voice", voice,
        "--output", str(output),
        "--speed", str(speed),
    ]


# ##################################################################
# make temp output path
# generates a unique temporary file path for the synthesised audio.
# the file is created and immediately closed so the subprocess can
# write to it without collision.
def _make_temp_output() -> Path:
    tmp = tempfile.NamedTemporaryFile(
        suffix=_DEFAULT_SUFFIX,
        prefix="agent_sdk_tts_",
        delete=False,
    )
    tmp.close()
    return Path(tmp.name)


# ##################################################################
# run subprocess
# runs a subprocess using asyncio.create_subprocess_exec and waits
# for it to complete. raises AgentError if the process exits non-zero
# or the timeout is exceeded.
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
# synthesize speech
# converts text to speech audio using the local tts subprocess tool.
# auto-generates an output path if not provided. uses _DEFAULT_VOICE
# if voice is not specified. speed defaults to 1.0 (normal speed).
async def synthesize_speech(
    text: str,
    *,
    voice: str = _DEFAULT_VOICE,
    output: str | Path | None = None,
    speed: float = 1.0,
    timeout: float = 120.0,
    logger: ConversationLogger | None = None,
    conversation_id: UUID | None = None,
) -> AudioResult:
    output_path = Path(output) if output is not None else _make_temp_output()
    conv_id = conversation_id or uuid.uuid4()

    if logger is not None:
        logger.log_event(
            "tts_request",
            text_length=len(text),
            voice=voice,
            output=str(output_path),
            speed=speed,
        )

    cmd = _build_tts_command(text, voice=voice, output=output_path, speed=speed)
    await _run_subprocess(cmd, timeout=timeout, label="tts")

    if logger is not None:
        logger.log_event("tts_complete", path=str(output_path), voice=voice)

    return AudioResult(
        path=output_path,
        model_used=_LOCAL_TTS_MODEL,
        conversation_id=conv_id,
        text=text,
        voice=voice,
    )
