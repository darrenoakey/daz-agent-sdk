from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


def _run_cli(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    entrypoint = Path(sys.executable).with_name("daz-agent-sdk")
    assert entrypoint.is_file()
    return subprocess.run(
        [str(entrypoint), *arguments], capture_output=True, text=True, check=False
    )


def _assert_concise_error(
    result: subprocess.CompletedProcess[str], expected: tuple[str, ...]
) -> None:
    assert result.returncode != 0
    assert result.stdout == ""
    assert result.stderr.startswith("Error: ")
    assert len(result.stderr.splitlines()) == 1
    assert "Traceback" not in result.stderr
    for text in expected:
        assert text in result.stderr


@pytest.mark.parametrize(
    "provider", ["flux", "z-image-turbo", "ollama", "gemini", "spark", "arbiter"]
)
@pytest.mark.parametrize("branch", ["generate", "state", "recover", "idempotency"])
def test_image_cli_rejects_provider_pins_before_artifacts(
    tmp_path: Path, provider: str, branch: str
) -> None:
    output = tmp_path / "result.png"
    state = tmp_path / "operation.json"
    arguments = [
        "image",
        "--provider",
        provider,
        "--output",
        str(output),
    ]
    if branch == "recover":
        arguments.extend(["--recover", str(state)])
    else:
        arguments.extend(
            ["--prompt", "route validation", "--width", "64", "--height", "64"]
        )
        if branch == "state":
            arguments.extend(["--state", str(state)])
        elif branch == "idempotency":
            arguments.extend(["--idempotency-key", "route-validation-key"])

    result = _run_cli(arguments)

    _assert_concise_error(
        result, (provider, "actively disabled", "Mac mini Codex image service")
    )
    assert not output.exists()
    assert not state.exists()


@pytest.mark.parametrize(
    ("option", "value", "guidance"),
    [
        ("--model", "flux-schnell", "owns model selection"),
        ("--steps", "4", "owns inference settings"),
        ("--timeout", "5", "operations wait indefinitely"),
    ],
)
def test_image_cli_rejects_compatibility_controls_before_artifacts(
    tmp_path: Path, option: str, value: str, guidance: str
) -> None:
    output = tmp_path / "result.png"
    state = tmp_path / "operation.json"

    result = _run_cli(
        [
            "image",
            "--prompt",
            "compatibility validation",
            "--width",
            "64",
            "--height",
            "64",
            "--output",
            str(output),
            "--state",
            str(state),
            option,
            value,
        ]
    )

    _assert_concise_error(
        result, ("actively disabled", "Mac mini Codex image", guidance)
    )
    assert not output.exists()
    assert not state.exists()


def test_image_cli_reports_validation_error_without_traceback(tmp_path: Path) -> None:
    output = tmp_path / "result.png"
    state = tmp_path / "operation.json"

    result = _run_cli(
        [
            "image",
            "--width",
            "64",
            "--height",
            "64",
            "--output",
            str(output),
            "--state",
            str(state),
        ]
    )

    _assert_concise_error(result, ("--prompt is required",))
    assert not output.exists()
    assert not state.exists()


def test_image_cli_reports_execution_error_without_traceback(tmp_path: Path) -> None:
    output = tmp_path / "result.png"
    state = tmp_path / "operation.json"
    missing_input = tmp_path / "missing.png"

    result = _run_cli(
        [
            "image",
            "--prompt",
            "edit validation",
            "--width",
            "64",
            "--height",
            "64",
            "--image",
            str(missing_input),
            "--output",
            str(output),
            "--state",
            str(state),
        ]
    )

    _assert_concise_error(result, ("Input image not found", str(missing_input)))
    assert not output.exists()
    assert not state.exists()
