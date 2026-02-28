from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import UUID

from daz_agent_sdk.logging_ import ConversationLogger
from daz_agent_sdk.types import AgentError, Capability, ErrorKind, ModelInfo, Tier


# ##################################################################
# local stt model info
# placeholder ModelInfo for the local whisper subprocess tool.
_LOCAL_STT_MODEL = ModelInfo(
    provider="local",
    model_id="whisper",
    display_name="Local Whisper STT",
    capabilities=frozenset({Capability.STT}),
    tier=Tier.HIGH,
    supports_streaming=False,
    supports_structured=False,
    supports_conversation=False,
)

# ##################################################################
# default model size
# whisper model size when caller does not specify
_DEFAULT_MODEL_SIZE = "small"


# ##################################################################
# build command
# constructs the whisper subprocess argument list from the supplied
# parameters. audio_path must already be resolved by the caller.
def _build_stt_command(
    audio_path: Path,
    *,
    model_size: str,
    language: str | None,
) -> list[str]:
    cmd = [
        "whisper",
        str(audio_path),
        "--model", model_size,
        "--output_format", "txt",
    ]
    if language is not None:
        cmd.extend(["--language", language])
    return cmd


# ##################################################################
# run subprocess
# runs a subprocess using asyncio.create_subprocess_exec and waits
# for it to complete. returns stdout text. raises AgentError if the
# process exits non-zero or the timeout is exceeded.
async def _run_subprocess(args: list[str], *, timeout: float, label: str) -> str:
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
        return stdout.decode(errors="replace")
    except AgentError:
        raise
    except OSError as exc:
        raise AgentError(
            f"{label} could not be started: {exc}",
            kind=ErrorKind.NOT_AVAILABLE,
        ) from exc


# ##################################################################
# transcribe
# transcribes audio to text using the local whisper subprocess tool.
# model_size controls the whisper model variant (base, small, large-v3-turbo).
# language is optional — whisper auto-detects if not provided.
async def transcribe(
    audio: str | Path,
    *,
    model_size: str = _DEFAULT_MODEL_SIZE,
    language: str | None = None,
    timeout: float = 120.0,
    logger: ConversationLogger | None = None,
    conversation_id: UUID | None = None,
) -> str:
    audio_path = Path(audio)

    if not audio_path.exists():
        raise AgentError(
            f"Audio file does not exist: {audio_path}",
            kind=ErrorKind.INVALID_REQUEST,
        )

    if logger is not None:
        logger.log_event(
            "stt_request",
            audio=str(audio_path),
            model_size=model_size,
            language=language,
        )

    cmd = _build_stt_command(audio_path, model_size=model_size, language=language)
    stdout = await _run_subprocess(cmd, timeout=timeout, label="whisper")
    text = stdout.strip()

    if logger is not None:
        logger.log_event(
            "stt_complete",
            audio=str(audio_path),
            text_length=len(text),
        )

    return text
