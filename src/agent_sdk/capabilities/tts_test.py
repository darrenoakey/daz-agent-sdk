from __future__ import annotations

from pathlib import Path

from agent_sdk.capabilities.tts import (
    _DEFAULT_VOICE,
    _build_tts_command,
    _make_temp_output,
)


# ##################################################################
# command building tests


def test_build_tts_command_structure():
    cmd = _build_tts_command(
        "Hello world",
        voice="gary",
        output=Path("/tmp/out.wav"),
        speed=1.0,
    )
    assert cmd[0] == "tts"
    assert cmd[1] == "tts"
    assert "--text" in cmd
    assert cmd[cmd.index("--text") + 1] == "Hello world"


def test_build_tts_command_voice_flag():
    cmd = _build_tts_command(
        "Test",
        voice="gary",
        output=Path("/tmp/out.wav"),
        speed=1.0,
    )
    assert "--voice" in cmd
    assert cmd[cmd.index("--voice") + 1] == "gary"


def test_build_tts_command_output_flag():
    out = Path("/tmp/speech_output.wav")
    cmd = _build_tts_command(
        "Test",
        voice="aiden",
        output=out,
        speed=1.0,
    )
    assert "--output" in cmd
    assert cmd[cmd.index("--output") + 1] == str(out)


def test_build_tts_command_speed_flag():
    cmd = _build_tts_command(
        "Test",
        voice="aiden",
        output=Path("/tmp/out.wav"),
        speed=1.5,
    )
    assert "--speed" in cmd
    assert cmd[cmd.index("--speed") + 1] == "1.5"


def test_build_tts_command_speed_default():
    cmd = _build_tts_command(
        "Test",
        voice="aiden",
        output=Path("/tmp/out.wav"),
        speed=1.0,
    )
    assert cmd[cmd.index("--speed") + 1] == "1.0"


# ##################################################################
# default voice tests


def test_default_voice_is_aiden():
    assert _DEFAULT_VOICE == "aiden"


def test_build_tts_uses_default_voice():
    cmd = _build_tts_command(
        "Test",
        voice=_DEFAULT_VOICE,
        output=Path("/tmp/out.wav"),
        speed=1.0,
    )
    assert cmd[cmd.index("--voice") + 1] == _DEFAULT_VOICE


# ##################################################################
# auto output path generation tests


def test_make_temp_output_returns_path():
    path = _make_temp_output()
    assert isinstance(path, Path)


def test_make_temp_output_has_wav_suffix():
    path = _make_temp_output()
    assert path.suffix == ".wav"


def test_make_temp_output_is_unique():
    p1 = _make_temp_output()
    p2 = _make_temp_output()
    assert p1 != p2


def test_make_temp_output_file_exists():
    # temp file should be created (not just a random path)
    path = _make_temp_output()
    assert path.exists()
    path.unlink(missing_ok=True)
