from __future__ import annotations

import asyncio
import base64
import inspect
import json
import subprocess
import sys
import uuid
from pathlib import Path

import pytest
from PIL import Image

import daz_agent_sdk.capabilities.image as image_module
from daz_agent_sdk import Agent
from daz_agent_sdk.capabilities.image import (
    download_image_job,
    generate_image,
    get_image_job,
    resume_image_job,
    resume_image_operation,
    submit_image_job,
)
from daz_agent_sdk.config import Config, ImageConfig, ImageTierConfig
from daz_agent_sdk.types import AgentError, ErrorKind


def test_image_service_is_hard_pinned_to_canonical_macmini_origin():
    source = inspect.getsource(image_module)
    assert source.count('"http://10.0.0.46:8830"') == 1
    assert 'arguments.append("http://10.0.0.46:8830" + path)' in source
    for function in vars(image_module).values():
        if callable(function):
            assert "origin" not in inspect.signature(function).parameters
    for forbidden in (
        "_generate_image_at",
        "_submit_image_job_at",
        "_get_image_job_at",
        "_resume_image_operation_at",
    ):
        assert not hasattr(image_module, forbidden)


def test_python_transport_is_proxy_and_redirect_immune():
    source = inspect.getsource(image_module._curl_image_service)
    for argument in (
        '"--proxy"',
        '"--noproxy"',
        '"--proto"',
        '"--proto-redir"',
        '"--max-redirs"',
    ):
        assert argument in source


def test_public_image_apis_reject_caller_controlled_service_url():
    for function in (
        generate_image,
        get_image_job,
        resume_image_job,
        download_image_job,
    ):
        assert "service_url" not in inspect.signature(function).parameters


@pytest.mark.parametrize(
    "provider",
    [
        "spark",
        "arbiter",
        "flux",
        "mflux",
        "z-image-turbo",
        "ollama",
        "gemini",
        "nano-banana-2",
        "openai",
    ],
)
def test_rejects_every_legacy_provider(provider: str):
    with pytest.raises(AgentError) as exc_info:
        asyncio.run(
            generate_image(
                "test", width=512, height=512, provider=provider, timeout=1.0
            )
        )
    assert provider in str(exc_info.value)
    assert "actively disabled" in str(exc_info.value)
    assert exc_info.value.kind == ErrorKind.INVALID_REQUEST


@pytest.mark.parametrize(
    "model",
    ["flux-schnell", "z-image-turbo", "gemini-3.1-flash-image-preview", "gpt-image-1"],
)
def test_rejects_every_legacy_model_pin(model: str):
    with pytest.raises(AgentError) as exc_info:
        asyncio.run(generate_image("test", width=512, height=512, model=model))
    assert "owns model selection" in str(exc_info.value)
    assert exc_info.value.kind == ErrorKind.INVALID_REQUEST


def test_rejects_image_step_pin():
    with pytest.raises(AgentError) as exc_info:
        asyncio.run(generate_image("test", width=512, height=512, steps=4))
    assert "owns inference settings" in str(exc_info.value)


@pytest.mark.parametrize(
    "legacy_image",
    [
        ImageConfig(model="old-model"),
        ImageConfig(codex_model="old-codex-model"),
        ImageConfig(tiers={"high": ImageTierConfig(steps=4)}),
        ImageConfig(fallback=["spark"]),
        ImageConfig(transparent_post_process="arbiter"),
    ],
)
def test_public_image_entrypoints_reject_legacy_config_before_io(
    tmp_path: Path, legacy_image: ImageConfig
):
    config = Config(image=legacy_image)
    output = tmp_path / "missing" / "output.png"
    missing_input = tmp_path / "missing-input.png"

    async def exercise() -> None:
        from daz_agent_sdk.core import Agent

        calls = [
            generate_image(
                "legacy config",
                width=64,
                height=64,
                image=missing_input,
                output=output,
                config=config,
            ),
            submit_image_job(
                "legacy config",
                width=64,
                height=64,
                image=missing_input,
                idempotency_key="durable-key",
                config=config,
            ),
            get_image_job("invalid/job", config=config),
            resume_image_job("job", output, timeout=0.0, config=config),
            download_image_job("job", output, config=config),
            resume_image_operation(tmp_path / "missing.json", config=config),
            Agent(config).remove_background(missing_input),
        ]
        for call in calls:
            with pytest.raises(AgentError, match="legacy image configuration") as error:
                await call
            assert error.value.kind == ErrorKind.INVALID_REQUEST
            assert "Mac mini Codex image service" in str(error.value)

    asyncio.run(exercise())
    assert not output.parent.exists()


def test_codex_is_the_default_provider():
    image_module._validate_image_route("test", 512, 512, None, None, None)
    assert image_module._CODEX_MODEL.provider == "codex"


def test_remove_background_fails_closed_with_durable_service_guidance():
    with pytest.raises(AgentError) as error:
        asyncio.run(Agent().remove_background("image.png"))
    assert error.value.kind == ErrorKind.INVALID_REQUEST
    assert "actively disabled" in str(error.value)
    assert "durable Mac mini Codex image service" in str(error.value)
    assert image_module._CODEX_MODEL.model_id == "macmini-image-service"


def test_codex_encodes_input_image_for_service(tmp_path: Path):
    image_data = b"real input bytes"
    input_path = tmp_path / "source.bin"
    input_path.write_bytes(image_data)
    assert image_module._service_input_images(input_path) == [
        base64.b64encode(image_data).decode("ascii")
    ]


def test_terminal_job_error_preserves_identity_and_never_recovers():
    status = image_module._status_from_service(
        {"id": "terminal-job", "status": "failed", "error": "service failure"},
        "terminal-job",
    )
    error = image_module._terminal_job_error(status)
    assert error.attempts == [
        {"job_id": "terminal-job", "status": "failed", "recoverable": False}
    ]


def test_transient_status_and_artifact_errors_are_recoverable():
    transient = AgentError("temporary GET failure", kind=ErrorKind.NOT_AVAILABLE)
    terminal = AgentError("terminal GET failure", kind=ErrorKind.INTERNAL)
    assert image_module._is_transient_image_error(transient) is True
    assert image_module._is_transient_image_error(terminal) is False


def test_submit_image_job_requires_caller_key_before_network():
    with pytest.raises(AgentError) as error:
        asyncio.run(
            submit_image_job(
                "required key proof", width=64, height=64, idempotency_key=""
            )
        )
    assert error.value.kind == ErrorKind.INVALID_REQUEST
    assert "required" in str(error.value)


def test_submission_parser_preserves_replay_and_conflict_identity():
    key = str(uuid.uuid4())
    accepted = json.dumps(
        {"id": "durable-job", "idempotency_key": key, "replayed": True}
    ).encode()
    replay = image_module._parse_image_submission(accepted, 202, key)
    assert replay.job_id == "durable-job"
    assert replay.idempotency_key == key
    assert replay.replayed is True
    conflict_body = json.dumps(
        {
            "id": "durable-job",
            "code": "idempotency_conflict",
            "error": "conflict",
        }
    ).encode()
    with pytest.raises(AgentError) as conflict:
        image_module._parse_image_submission(conflict_body, 409, key)
    assert conflict.value.kind == ErrorKind.INVALID_REQUEST
    assert conflict.value.attempts == [
        {
            "idempotency_key": key,
            "job_id": "durable-job",
            "status": 409,
            "code": "idempotency_conflict",
            "recoverable": False,
        }
    ]


def test_python_durable_state_preserves_exact_submission_body_and_key(tmp_path: Path):
    key = str(uuid.uuid4())
    body = '{"prompt":"exact bytes","width":64,"height":64}'
    state_path = tmp_path / "image.state.json"
    _, state = image_module._prepare_operation(
        body.encode(), tmp_path / "out.png", False, key, state_path
    )
    persisted = image_module._read_operation(state_path)
    assert persisted["request_body"] == body
    assert persisted["idempotency_key"] == key
    assert persisted == state


def test_python_durable_state_rejects_mode_schema_and_immutable_metadata(
    tmp_path: Path,
):
    state_path = tmp_path / "image.state.json"
    body = b'{"prompt":"exact","width":64,"height":64}'
    _, state = image_module._prepare_operation(
        body, tmp_path / "out.png", False, "durable-key", state_path
    )
    state_path.chmod(0o640)
    with pytest.raises(PermissionError):
        image_module._read_operation(state_path)
    state_path.chmod(0o600)
    for field, value in (
        ("operation_id", "tampered"),
        ("output_path", str(tmp_path / "different.png")),
        ("unknown", "field"),
    ):
        changed = dict(state)
        changed[field] = value
        image_module._atomic_write_operation(state_path, changed)
        with pytest.raises(ValueError):
            image_module._read_operation(state_path)


def test_state_parent_symlink_is_rejected_before_creating_beneath_it(
    tmp_path: Path,
):
    owned = tmp_path / "owned"
    owned.mkdir(mode=0o700)
    linked = tmp_path / "linked"
    linked.symlink_to(owned, target_is_directory=True)
    state_path = linked / "missing" / "operation.json"
    with pytest.raises(OSError):
        image_module._prepare_operation(
            b'{"prompt":"symlink proof","width":64,"height":64}',
            tmp_path / "out.png",
            False,
            None,
            state_path,
        )
    assert not (owned / "missing").exists()


def test_high_level_image_defaults_have_no_finite_completion_cutoff():
    from daz_agent_sdk.core import Agent

    assert inspect.signature(generate_image).parameters["timeout"].default is None
    assert inspect.signature(resume_image_job).parameters["timeout"].default is None
    assert (
        inspect.signature(resume_image_operation).parameters["timeout"].default is None
    )
    assert inspect.signature(Agent.image).parameters["timeout"].default is None
    assert (
        inspect.signature(Agent.resume_image_job).parameters["timeout"].default is None
    )
    assert (
        inspect.signature(Agent.resume_image_operation).parameters["timeout"].default
        is None
    )


def test_public_image_timeouts_fail_before_service_or_filesystem_io(tmp_path: Path):
    from daz_agent_sdk.core import Agent

    output = tmp_path / "missing" / "output.png"
    state = tmp_path / "missing-operation.json"

    async def exercise() -> None:
        calls = [
            generate_image("deadline", width=64, height=64, output=output, timeout=1.0),
            resume_image_job("durable-job", output, timeout=1.0),
            resume_image_operation(state, timeout=1.0),
            Agent().image("deadline", width=64, height=64, output=output, timeout=1.0),
            Agent().resume_image_job("durable-job", output=output, timeout=1.0),
            Agent().resume_image_operation(state, timeout=1.0),
        ]
        for call in calls:
            with pytest.raises(
                AgentError, match="deadlines are actively disabled"
            ) as error:
                await call
            assert error.value.kind == ErrorKind.INVALID_REQUEST
            assert "wait indefinitely" in str(error.value)

    asyncio.run(exercise())
    assert not output.parent.exists()
    assert not state.exists()


def test_default_api_operation_survives_process_boundary_before_acceptance(
    tmp_path: Path,
):
    state_path = tmp_path / "operation.json"
    output = tmp_path / "result.png"
    body = '{"prompt":"restart proof","width":64,"height":64,"transparent":false}'
    script = (
        "import json,sys;"
        "from daz_agent_sdk.capabilities.image import _prepare_operation;"
        "p,s=_prepare_operation(sys.argv[1].encode(),sys.argv[2],False,None,sys.argv[3]);"
        "print(json.dumps({'path':str(p),'state':s},sort_keys=True))"
    )
    arguments = [sys.executable, "-c", script, body, str(output), str(state_path)]
    first = subprocess.run(arguments, capture_output=True, text=True, check=True)
    second = subprocess.run(arguments, capture_output=True, text=True, check=True)
    first_state = json.loads(first.stdout)["state"]
    second_state = json.loads(second.stdout)["state"]
    assert first_state == second_state
    assert first_state["job_id"] == ""
    assert first_state["request_body"] == body
    assert first_state["output_path"] == str(output.resolve())
    assert first_state["idempotency_key"]


def test_explicit_key_is_required_for_deliberate_regeneration_identity(tmp_path: Path):
    body = b'{"prompt":"same","width":64,"height":64,"transparent":false}'
    first_path, first = image_module._prepare_operation(
        body, tmp_path / "out.png", False, "operation-one", tmp_path / "one.json"
    )
    second_path, second = image_module._prepare_operation(
        body, tmp_path / "out.png", False, "operation-two", tmp_path / "two.json"
    )
    assert first_path != second_path
    assert first["operation_id"] != second["operation_id"]
    assert first["request_body"] == second["request_body"] == body.decode()


@pytest.mark.parametrize(
    ("status", "code"),
    [(409, "idempotency_conflict"), (410, "idempotency_expired")],
)
def test_python_transport_classifies_non_json_terminal_identity_errors(
    status: int, code: str
):
    key = str(uuid.uuid4())
    captured = image_module._terminal_submission_error(
        status, b"legacy terminal response", key
    )
    assert captured.kind == ErrorKind.INVALID_REQUEST
    assert captured.attempts == [
        {
            "idempotency_key": key,
            "job_id": "",
            "status": status,
            "code": code,
            "recoverable": False,
        }
    ]


def test_python_operation_state_and_lock_reject_symlinks_and_use_private_modes(
    tmp_path: Path,
):
    state_path = tmp_path / "state.json"
    target = tmp_path / "target"
    target.write_text("unchanged", encoding="utf-8")
    state_path.symlink_to(target)
    with pytest.raises(OSError):
        image_module._read_operation(state_path)
    state_path.unlink()
    lock_path = tmp_path / "state.json.lock"
    lock_path.symlink_to(target)
    with pytest.raises(OSError):
        asyncio.run(_enter_operation_lock(state_path))
    assert target.read_text(encoding="utf-8") == "unchanged"
    lock_path.unlink()
    image_module._atomic_write_operation(state_path, {"version": 2})
    asyncio.run(_assert_operation_lock_modes(state_path, lock_path))


async def _enter_operation_lock(state_path: Path) -> None:
    async with image_module._operation_lock(state_path):
        pass


async def _assert_operation_lock_modes(state_path: Path, lock_path: Path) -> None:
    async with image_module._operation_lock(state_path):
        assert state_path.stat().st_mode & 0o777 == 0o600
        assert lock_path.stat().st_mode & 0o777 == 0o600


def test_cancelled_lock_waiter_does_not_release_another_owner(tmp_path: Path):
    async def exercise() -> None:
        state_path = tmp_path / "cancel.json"
        owner_entered = asyncio.Event()
        release_owner = asyncio.Event()
        probe_entered = asyncio.Event()

        async def owner() -> None:
            async with image_module._operation_lock(state_path):
                owner_entered.set()
                await release_owner.wait()

        async def waiter(entered: asyncio.Event | None = None) -> None:
            async with image_module._operation_lock(state_path):
                if entered is not None:
                    entered.set()

        owner_task = asyncio.create_task(owner())
        await owner_entered.wait()
        cancelled_waiter = asyncio.create_task(waiter())
        await _next_event_loop_turn()
        cancelled_waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled_waiter
        probe = asyncio.create_task(waiter(probe_entered))
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(probe_entered.wait(), timeout=0.05)
        release_owner.set()
        await asyncio.gather(owner_task, probe)
        assert probe_entered.is_set()

    asyncio.run(exercise())


def test_cancelled_lock_owner_releases_its_owned_lock(tmp_path: Path):
    async def exercise() -> None:
        state_path = tmp_path / "cancel-owner.json"
        entered = asyncio.Event()
        hold = asyncio.Event()

        async def owner() -> None:
            async with image_module._operation_lock(state_path):
                entered.set()
                await hold.wait()

        task = asyncio.create_task(owner())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.wait_for(_enter_operation_lock(state_path), timeout=1.0)

    asyncio.run(exercise())


async def _next_event_loop_turn() -> None:
    loop = asyncio.get_running_loop()
    turn = loop.create_future()
    loop.call_soon(turn.set_result, None)
    await turn


def test_cross_process_lock_wait_keeps_event_loop_live(tmp_path: Path):
    state_path = tmp_path / "cross-process.json"
    script = (
        "import sys;"
        "from pathlib import Path;"
        "from daz_agent_sdk.capabilities.image import "
        "_open_operation_lock,_release_operation_lock,_try_operation_lock;"
        "descriptor=_open_operation_lock(Path(sys.argv[1]));"
        "assert _try_operation_lock(descriptor);"
        "print('owned',flush=True);"
        "sys.stdin.read(1);"
        "_release_operation_lock(descriptor,True)"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(state_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "owned"

        async def contend() -> None:
            entered = asyncio.Event()

            async def acquire() -> None:
                async with image_module._operation_lock(state_path):
                    entered.set()

            contender = asyncio.create_task(acquire())
            await _next_event_loop_turn()
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(entered.wait(), timeout=0.05)
            assert process.stdin is not None
            process.stdin.write("x")
            process.stdin.flush()
            await asyncio.wait_for(contender, timeout=5.0)
            assert entered.is_set()

        asyncio.run(contend())
        assert process.wait(timeout=5.0) == 0
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5.0)
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                stream.close()


def test_python_operation_rejects_state_in_writable_shared_directory(tmp_path: Path):
    shared = tmp_path / "shared"
    shared.mkdir(mode=0o777)
    shared.chmod(0o777)
    with pytest.raises(PermissionError, match="owner-only"):
        image_module._atomic_write_operation(shared / "state.json", {"version": 2})


def test_python_operation_atomic_state_replace_rejects_symlink(tmp_path: Path):
    state_path = tmp_path / "state.json"
    target = tmp_path / "target"
    target.write_text("unchanged", encoding="utf-8")
    state_path.symlink_to(target)
    with pytest.raises(PermissionError, match="unsafe"):
        image_module._atomic_write_operation(state_path, {"version": 2})
    assert target.read_text(encoding="utf-8") == "unchanged"
    assert state_path.is_symlink()


def test_input_image_missing_raises():
    with pytest.raises(AgentError) as exc_info:
        asyncio.run(
            generate_image(
                "edit this",
                width=512,
                height=512,
                image="/nonexistent/does-not-exist.png",
            )
        )
    assert exc_info.value.kind == ErrorKind.INVALID_REQUEST


@pytest.mark.real_igs
def test_generate_real_image(tmp_path: Path):
    output = tmp_path / "igs-canary.png"
    state = tmp_path / "igs-canary.state.json"
    result = asyncio.run(
        Agent().image(
            "A cheerful cartoon robot waving hello, centered on a clean white background",
            width=256,
            height=256,
            output=output,
            operation_state=state,
            idempotency_key=str(uuid.uuid4()),
        )
    )
    assert result.path == output
    assert result.path.stat().st_size > 1000
    assert result.width == 256
    assert result.height == 256
    assert result.model_used.provider == "codex"
    assert result.model_used.model_id == "macmini-image-service"
    assert result.job_id
    assert result.status == "done"
    assert result.ready is True
    assert result.path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert state.is_file()
    with Image.open(result.path) as image:
        assert image.format == "PNG"
        assert image.size == (256, 256)
        image.verify()
    print(
        f"IGS canary job={result.job_id} provider={result.model_used.provider} artifact={result.path}"
    )
