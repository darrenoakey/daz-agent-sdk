from __future__ import annotations

from pathlib import Path

import pytest

from agent_sdk.capabilities.stt import (
    _DEFAULT_MODEL_SIZE,
    _LOCAL_STT_MODEL,
    _build_stt_command,
)
from agent_sdk.types import AgentError, Capability


# ##################################################################
# build command tests
# verify the argument list structure for whisper subprocess calls
def test_build_stt_command_structure() -> None:
    cmd = _build_stt_command(Path("/tmp/audio.wav"), model_size="small", language=None)
    assert cmd[0] == "whisper"
    assert "/tmp/audio.wav" in cmd


def test_build_stt_command_model_flag() -> None:
    cmd = _build_stt_command(Path("/tmp/a.wav"), model_size="large-v3-turbo", language=None)
    idx = cmd.index("--model")
    assert cmd[idx + 1] == "large-v3-turbo"


def test_build_stt_command_output_format() -> None:
    cmd = _build_stt_command(Path("/tmp/a.wav"), model_size="small", language=None)
    idx = cmd.index("--output_format")
    assert cmd[idx + 1] == "txt"


def test_build_stt_command_language_flag() -> None:
    cmd = _build_stt_command(Path("/tmp/a.wav"), model_size="small", language="en")
    idx = cmd.index("--language")
    assert cmd[idx + 1] == "en"


def test_build_stt_command_no_language_flag() -> None:
    cmd = _build_stt_command(Path("/tmp/a.wav"), model_size="small", language=None)
    assert "--language" not in cmd


# ##################################################################
# default model size
# verify the default whisper model size constant
def test_default_model_size() -> None:
    assert _DEFAULT_MODEL_SIZE == "small"


# ##################################################################
# local stt model info
# verify the placeholder model has correct capability
def test_local_stt_model_has_stt_capability() -> None:
    assert Capability.STT in _LOCAL_STT_MODEL.capabilities


def test_local_stt_model_provider() -> None:
    assert _LOCAL_STT_MODEL.provider == "local"


# ##################################################################
# transcribe validation
# verify that transcribe raises on missing audio file
@pytest.mark.asyncio
async def test_transcribe_missing_file_raises(tmp_path: Path) -> None:
    from agent_sdk.capabilities.stt import transcribe
    missing = tmp_path / "nonexistent.wav"
    with pytest.raises(AgentError) as exc_info:
        await transcribe(missing)
    assert exc_info.value.kind.value == "invalid_request"
