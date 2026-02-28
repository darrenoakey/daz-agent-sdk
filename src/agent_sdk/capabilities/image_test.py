from __future__ import annotations

from pathlib import Path

from agent_sdk.capabilities.image import (
    _build_command,
    _resolve_steps,
)
from agent_sdk.config import Config, ImageConfig, ImageTierConfig
from agent_sdk.types import Tier


# ##################################################################
# helpers
# build a minimal Config with only image tier steps configured


def _make_config(very_high: int = 32, high: int = 16, medium: int = 8, low: int = 2) -> Config:
    image = ImageConfig(
        model="z-image-turbo",
        tiers={
            "very_high": ImageTierConfig(steps=very_high),
            "high": ImageTierConfig(steps=high),
            "medium": ImageTierConfig(steps=medium),
            "low": ImageTierConfig(steps=low),
        },
    )
    cfg = Config(image=image)
    return cfg


# ##################################################################
# step resolution tests


def test_resolve_steps_very_high_tier():
    cfg = _make_config(very_high=32)
    assert _resolve_steps(Tier.VERY_HIGH, None, cfg) == 32


def test_resolve_steps_high_tier():
    cfg = _make_config(high=16)
    assert _resolve_steps(Tier.HIGH, None, cfg) == 16


def test_resolve_steps_medium_tier():
    cfg = _make_config(medium=8)
    assert _resolve_steps(Tier.MEDIUM, None, cfg) == 8


def test_resolve_steps_low_tier():
    cfg = _make_config(low=2)
    assert _resolve_steps(Tier.LOW, None, cfg) == 2


def test_resolve_steps_explicit_override():
    cfg = _make_config(high=8)
    # explicit steps should always win regardless of tier
    assert _resolve_steps(Tier.HIGH, 12, cfg) == 12


def test_resolve_steps_zero_explicit_override():
    cfg = _make_config(medium=4)
    # zero is a valid explicit override
    assert _resolve_steps(Tier.MEDIUM, 0, cfg) == 0


def test_resolve_steps_uses_default_config_when_none():
    # no config supplied — should fall back to load_config defaults
    steps = _resolve_steps(Tier.HIGH, None, None)
    assert isinstance(steps, int)
    assert steps > 0


# ##################################################################
# command building tests


def test_build_command_basic():
    cmd = _build_command(
        "a robot in the rain",
        width=512,
        height=512,
        output=Path("/tmp/out.jpg"),
        model="z-image-turbo",
        steps=4,
    )
    assert cmd[0] == "generate_image"
    assert "--prompt" in cmd
    idx = cmd.index("--prompt")
    assert cmd[idx + 1] == "a robot in the rain"


def test_build_command_includes_width_height():
    cmd = _build_command(
        "test",
        width=1024,
        height=768,
        output=Path("/tmp/out.jpg"),
        model="z-image-turbo",
        steps=8,
    )
    assert "--width" in cmd
    assert cmd[cmd.index("--width") + 1] == "1024"
    assert "--height" in cmd
    assert cmd[cmd.index("--height") + 1] == "768"


def test_build_command_includes_output():
    out = Path("/tmp/test_image.png")
    cmd = _build_command(
        "test",
        width=512,
        height=512,
        output=out,
        model="z-image-turbo",
        steps=4,
    )
    assert "--output" in cmd
    assert cmd[cmd.index("--output") + 1] == str(out)


def test_build_command_includes_model():
    cmd = _build_command(
        "test",
        width=512,
        height=512,
        output=Path("/tmp/out.jpg"),
        model="custom-model",
        steps=4,
    )
    assert "--model" in cmd
    assert cmd[cmd.index("--model") + 1] == "custom-model"


def test_build_command_includes_steps():
    cmd = _build_command(
        "test",
        width=512,
        height=512,
        output=Path("/tmp/out.jpg"),
        model="z-image-turbo",
        steps=12,
    )
    assert "--steps" in cmd
    assert cmd[cmd.index("--steps") + 1] == "12"


def test_build_command_does_not_include_transparent_when_false():
    cmd = _build_command(
        "test",
        width=512,
        height=512,
        output=Path("/tmp/out.jpg"),
        model="z-image-turbo",
        steps=4,
    )
    assert "--transparent" not in cmd


def test_build_command_includes_source_image():
    cmd = _build_command(
        "enhance this",
        width=512,
        height=512,
        output=Path("/tmp/out.jpg"),
        model="z-image-turbo",
        steps=4,
        image=Path("/tmp/source.jpg"),
    )
    assert "--image" in cmd
    assert cmd[cmd.index("--image") + 1] == "/tmp/source.jpg"


def test_build_command_includes_image_strength():
    cmd = _build_command(
        "enhance this",
        width=512,
        height=512,
        output=Path("/tmp/out.jpg"),
        model="z-image-turbo",
        steps=4,
        image=Path("/tmp/source.jpg"),
        image_strength=0.7,
    )
    assert "--image-strength" in cmd
    assert cmd[cmd.index("--image-strength") + 1] == "0.7"


def test_build_command_includes_guidance():
    cmd = _build_command(
        "test",
        width=512,
        height=512,
        output=Path("/tmp/out.jpg"),
        model="z-image-turbo",
        steps=4,
        guidance=7.5,
    )
    assert "--guidance" in cmd
    assert cmd[cmd.index("--guidance") + 1] == "7.5"


def test_build_command_includes_quantize():
    cmd = _build_command(
        "test",
        width=512,
        height=512,
        output=Path("/tmp/out.jpg"),
        model="z-image-turbo",
        steps=4,
        quantize=4,
    )
    assert "--quantize" in cmd
    assert cmd[cmd.index("--quantize") + 1] == "4"


def test_build_command_includes_seed():
    cmd = _build_command(
        "test",
        width=512,
        height=512,
        output=Path("/tmp/out.jpg"),
        model="z-image-turbo",
        steps=4,
        seed=42,
    )
    assert "--seed" in cmd
    assert cmd[cmd.index("--seed") + 1] == "42"


def test_build_command_no_image_by_default():
    cmd = _build_command(
        "test",
        width=512,
        height=512,
        output=Path("/tmp/out.jpg"),
        model="z-image-turbo",
        steps=4,
    )
    assert "--image" not in cmd
    assert "--image-strength" not in cmd


def test_build_command_includes_transparent_when_true():
    cmd = _build_command(
        "test",
        width=512,
        height=512,
        output=Path("/tmp/out.png"),
        model="z-image-turbo",
        steps=4,
        transparent=True,
    )
    assert "--transparent" in cmd


# ##################################################################
# explicit steps override test


def test_explicit_steps_override_in_command():
    cmd = _build_command(
        "test",
        width=512,
        height=512,
        output=Path("/tmp/out.jpg"),
        model="z-image-turbo",
        steps=99,
    )
    assert cmd[cmd.index("--steps") + 1] == "99"


