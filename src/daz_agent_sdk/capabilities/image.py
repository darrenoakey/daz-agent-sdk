from __future__ import annotations

import asyncio
import base64
import fcntl
import hashlib
import json
import os
import struct
import stat
import subprocess
import uuid
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Any, AsyncIterator, Sequence
from uuid import UUID

from daz_agent_sdk.config import Config
from daz_agent_sdk.logging_ import ConversationLogger
from daz_agent_sdk.types import (
    AgentError,
    Capability,
    ErrorKind,
    ImageJobStatus,
    ImageResult,
    ImageSubmission,
    ModelInfo,
    Tier,
)


# ##################################################################
# codex model info — image generation always goes through the mac mini
# image_generation_service (HTTP at :8830), which is backed by ChatGPT's
# image_generation tool over the codex responses endpoint.
_CODEX_MODEL = ModelInfo(
    provider="codex",
    model_id="macmini-image-service",
    display_name="Mac mini image generation service",
    capabilities=frozenset({Capability.IMAGE}),
    tier=Tier.HIGH,
    supports_streaming=False,
    supports_structured=False,
    supports_conversation=False,
)

_SUBMISSION_ATTEMPTS = 3
_OPERATION_STATE_VERSION = 2
_OPERATION_NAMESPACE = "daz-agent-sdk:image-operation:"
_MAX_STATE_BYTES = 1 << 20
_MAX_IMAGE_BYTES = 128 << 20


def _validate_legacy_image_config(config: Config | None) -> None:
    if config is None:
        return
    image = config.image
    configured = []
    if image.model.strip():
        configured.append("model")
    if image.codex_model.strip():
        configured.append("codex_model")
    if image.tiers:
        configured.append("tiers")
    if image.fallback:
        configured.append("fallback")
    if image.transparent_post_process.strip():
        configured.append("transparent.post_process")
    if configured:
        fields = ", ".join(configured)
        raise AgentError(
            f"legacy image configuration is actively disabled: {fields}",
            kind=ErrorKind.INVALID_REQUEST,
        )


def _operation_registry_root() -> Path:
    home = Path.home().absolute()
    root = home / ".daz-agent-sdk" / "image-operations"
    descriptor = os.open(home, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for index, component in enumerate((".daz-agent-sdk", "image-operations")):
            try:
                os.mkdir(component, 0o700, dir_fd=descriptor)
            except FileExistsError:
                pass
            next_descriptor = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
            details = os.fstat(descriptor)
            required_mode = 0o700 if index == 1 else None
            if (
                details.st_uid != os.geteuid()
                or (
                    required_mode is not None
                    and stat.S_IMODE(details.st_mode) != required_mode
                )
                or stat.S_IMODE(details.st_mode) & 0o022
            ):
                raise PermissionError(
                    f"image operation registry is not private: {root}"
                )
    finally:
        os.close(descriptor)
    details = root.lstat()
    if (
        not stat.S_ISDIR(details.st_mode)
        or details.st_uid != os.geteuid()
        or stat.S_IMODE(details.st_mode) != 0o700
    ):
        raise PermissionError(f"image operation registry is not private: {root}")
    return root


def _operation_identity(body: bytes, output_intent: str, key: str | None) -> str:
    identity = (
        b"idempotency-key\0" + key.encode("utf-8")
        if key
        else b"request\0" + body + b"\0output\0" + output_intent.encode("utf-8")
    )
    return hashlib.sha256(identity).hexdigest()


def _operation_key(operation_id: str, requested_key: str | None) -> str:
    if requested_key:
        return requested_key
    return str(uuid.uuid5(uuid.NAMESPACE_URL, _OPERATION_NAMESPACE + operation_id))


def _operation_output_intent(output: str | Path | None) -> str:
    if output is None:
        return "automatic:png"
    return "path:" + str(Path(output).expanduser().absolute())


def _operation_output_path(operation_id: str, output: str | Path | None) -> Path:
    if output is not None:
        return Path(output).expanduser().absolute()
    artifact_root = _operation_registry_root() / "artifacts"
    descriptor = _open_or_create_directory(artifact_root, 0o700)
    os.close(descriptor)
    return artifact_root / f"{operation_id}.png"


def _operation_output_format(output_path: Path) -> str:
    return "jpeg" if output_path.suffix.lower() in {".jpg", ".jpeg"} else "png"


def _operation_state_path(operation_id: str, selected: str | Path | None) -> Path:
    if selected is not None:
        return Path(selected).expanduser().absolute()
    return _operation_registry_root() / f"{operation_id}.json"


def _atomic_write_operation(path: Path, state: dict[str, Any]) -> None:
    directory = _open_or_create_operation_directory(path.parent)
    try:
        existing = os.stat(path.name, dir_fd=directory, follow_symlinks=False)
    except FileNotFoundError:
        existing = None
    if existing is not None and not stat.S_ISREG(existing.st_mode):
        os.close(directory)
        raise PermissionError(f"image operation state target is unsafe: {path}")
    temporary = f".{path.name}.{uuid.uuid4()}"
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
        dir_fd=directory,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(state, stream, separators=(",", ":"), sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path.name, src_dir_fd=directory, dst_dir_fd=directory)
        os.fsync(directory)
    finally:
        try:
            os.unlink(temporary, dir_fd=directory)
        except FileNotFoundError:
            pass
        os.close(directory)


def _read_operation(path: Path) -> dict[str, Any]:
    _validate_operation_directory(path)
    directory = _open_parent_directory(path)
    descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
    os.close(directory)
    with os.fdopen(descriptor, encoding="utf-8") as stream:
        details = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(details.st_mode)
            or details.st_uid != os.geteuid()
            or stat.S_IMODE(details.st_mode) != 0o600
            or details.st_size > _MAX_STATE_BYTES
        ):
            raise PermissionError(
                f"image operation state is not a user-owned regular file: {path}"
            )
        raw = stream.read(_MAX_STATE_BYTES + 1)
    if len(raw.encode("utf-8")) > _MAX_STATE_BYTES:
        raise ValueError(f"image operation state exceeds size limit: {path}")
    state = json.loads(raw)
    if not isinstance(state, dict):
        raise ValueError(f"image operation state is not a JSON object: {path}")
    _validate_operation_state(state, path)
    return state


def _open_operation_lock(path: Path) -> int:
    lock_path = path.with_name(path.name + ".lock")
    directory = _open_or_create_operation_directory(lock_path.parent)
    try:
        try:
            descriptor = os.open(
                lock_path.name,
                os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
                dir_fd=directory,
            )
        except FileExistsError:
            descriptor = os.open(
                lock_path.name,
                os.O_RDWR | os.O_NOFOLLOW,
                dir_fd=directory,
            )
    finally:
        os.close(directory)
    details = os.fstat(descriptor)
    if (
        not stat.S_ISREG(details.st_mode)
        or details.st_uid != os.geteuid()
        or stat.S_IMODE(details.st_mode) != 0o600
    ):
        os.close(descriptor)
        raise PermissionError(f"image operation lock is not private: {lock_path}")
    return descriptor


def _try_operation_lock(descriptor: int) -> bool:
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    return True


def _release_operation_lock(descriptor: int, owned: bool) -> None:
    try:
        if owned:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


async def _open_operation_lock_async(path: Path) -> int:
    opening = asyncio.create_task(asyncio.to_thread(_open_operation_lock, path))
    try:
        return await asyncio.shield(opening)
    except asyncio.CancelledError:
        descriptor = await asyncio.shield(opening)
        await asyncio.shield(
            asyncio.to_thread(_release_operation_lock, descriptor, False)
        )
        raise


@asynccontextmanager
async def _operation_lock(path: Path) -> AsyncIterator[None]:
    descriptor = await _open_operation_lock_async(path)
    owned = False
    try:
        while not owned:
            attempt = asyncio.create_task(
                asyncio.to_thread(_try_operation_lock, descriptor)
            )
            try:
                owned = await asyncio.shield(attempt)
            except asyncio.CancelledError:
                owned = await asyncio.shield(attempt)
                await asyncio.shield(
                    asyncio.to_thread(_release_operation_lock, descriptor, owned)
                )
                descriptor = -1
                raise
            if not owned:
                await _wait_for_poll(0.01)
        yield
    finally:
        if descriptor >= 0:
            await asyncio.shield(
                asyncio.to_thread(_release_operation_lock, descriptor, owned)
            )


def _validate_operation_directory(path: Path) -> None:
    directory = _open_parent_directory(path)
    try:
        details = os.fstat(directory)
    finally:
        os.close(directory)
    if (
        not stat.S_ISDIR(details.st_mode)
        or details.st_uid != os.geteuid()
        or stat.S_IMODE(details.st_mode) != 0o700
    ):
        raise PermissionError(
            f"image operation directory must be a current-user owner-only directory: {path.parent}"
        )


def _reject_symlink_components(path: Path) -> None:
    current = Path(path.anchor)
    for component in path.absolute().parts[1:]:
        current /= component
        if stat.S_ISLNK(current.lstat().st_mode):
            raise PermissionError(f"symlink path component is forbidden: {current}")


def _open_parent_directory(path: Path) -> int:
    absolute = path.parent.absolute()
    descriptor = os.open(absolute.anchor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in absolute.parts[1:]:
            next_descriptor = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_or_create_directory(path: Path, mode: int) -> int:
    absolute = path.expanduser().absolute()
    descriptor = os.open(absolute.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for component in absolute.parts[1:]:
            try:
                next_descriptor = os.open(
                    component,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=descriptor,
                )
            except FileNotFoundError:
                os.mkdir(component, mode, dir_fd=descriptor)
                next_descriptor = os.open(
                    component,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=descriptor,
                )
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_or_create_operation_directory(path: Path) -> int:
    descriptor = _open_or_create_directory(path, 0o700)
    details = os.fstat(descriptor)
    if (
        details.st_uid != os.geteuid()
        or not stat.S_ISDIR(details.st_mode)
        or stat.S_IMODE(details.st_mode) != 0o700
    ):
        os.close(descriptor)
        raise PermissionError(
            f"image operation directory must be a current-user owner-only directory: {path}"
        )
    return descriptor


def _operation_state_exists(path: Path) -> bool:
    directory = _open_or_create_operation_directory(path.parent)
    try:
        try:
            details = os.stat(path.name, dir_fd=directory, follow_symlinks=False)
        except FileNotFoundError:
            return False
        if not stat.S_ISREG(details.st_mode):
            raise PermissionError(f"image operation state target is unsafe: {path}")
        return True
    finally:
        os.close(directory)


def _validate_operation_state(state: dict[str, Any], path: Path) -> None:
    expected = {
        "version": int,
        "operation_id": str,
        "idempotency_key": str,
        "request_body": str,
        "output_intent": str,
        "output_path": str,
        "output_format": str,
        "transparent": bool,
        "job_id": str,
    }
    if set(state) != set(expected) or any(
        type(state[name]) is not kind for name, kind in expected.items()
    ):
        raise ValueError(f"image operation state has invalid schema: {path}")
    body = state["request_body"]
    try:
        parsed = json.loads(body)
    except ValueError as exc:
        raise ValueError(f"image operation request body is invalid: {path}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"image operation request body is not an object: {path}")
    output_path = Path(state["output_path"])
    if not output_path.is_absolute():
        raise ValueError(f"image operation output path is not absolute: {path}")
    if state["output_format"] != _operation_output_format(output_path):
        raise ValueError(f"image operation output format conflicts with path: {path}")
    expected_intent = (
        "automatic:png"
        if state["output_intent"] == "automatic:png"
        else "path:" + str(output_path)
    )
    if state["output_intent"] != expected_intent:
        raise ValueError(f"image operation output metadata conflicts: {path}")
    expected_id = _operation_identity(
        body.encode("utf-8"), state["output_intent"], state["idempotency_key"]
    )
    keyed_id = _operation_identity(body.encode("utf-8"), state["output_intent"], None)
    if state["operation_id"] not in {expected_id, keyed_id}:
        raise ValueError(f"image operation immutable identity is invalid: {path}")


def _prepare_operation(
    body: bytes,
    output: str | Path | None,
    transparent: bool,
    idempotency_key: str | None,
    state_path: str | Path | None,
) -> tuple[Path, dict[str, Any]]:
    output_intent = _operation_output_intent(output)
    operation_id = _operation_identity(body, output_intent, idempotency_key)
    selected_path = _operation_state_path(operation_id, state_path)
    if _operation_state_exists(selected_path):
        state = _read_operation(selected_path)
        expected = (operation_id, body.decode("utf-8"), output_intent)
        actual = (
            state.get("operation_id"),
            state.get("request_body"),
            state.get("output_intent"),
        )
        if state.get("version") != _OPERATION_STATE_VERSION or actual != expected:
            raise AgentError(
                f"image operation identity conflicts with immutable state {selected_path}",
                kind=ErrorKind.INVALID_REQUEST,
            )
        return selected_path, state
    output_path = _operation_output_path(operation_id, output)
    state = {
        "version": _OPERATION_STATE_VERSION,
        "operation_id": operation_id,
        "idempotency_key": _operation_key(operation_id, idempotency_key),
        "request_body": body.decode("utf-8"),
        "output_intent": output_intent,
        "output_path": str(output_path),
        "output_format": _operation_output_format(output_path),
        "transparent": transparent,
        "job_id": "",
    }
    _atomic_write_operation(selected_path, state)
    return selected_path, state


# ##################################################################
# image service helpers


def _service_input_images(input_image: Path | Sequence[Path] | None) -> list[str]:
    """Encode input/reference images as base64 strings for the service."""
    if input_image is None:
        return []
    paths = list(input_image) if isinstance(input_image, Sequence) else [input_image]
    encoded: list[str] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            raise AgentError(
                f"Input image not found: {path}",
                kind=ErrorKind.INVALID_REQUEST,
            )
        data = path.read_bytes()
        if not data:
            raise AgentError(
                f"Input image is empty: {path}",
                kind=ErrorKind.INVALID_REQUEST,
            )
        encoded.append(base64.b64encode(data).decode("ascii"))
    return encoded


async def _service_json(
    method: str,
    path: str,
    *,
    payload: dict[str, Any] | None = None,
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    body = json.dumps(payload).encode() if payload is not None else None
    response, status, _ = await _curl_image_service(
        method, path, body, idempotency_key=idempotency_key
    )
    text = response.decode("utf-8", errors="replace")
    if status in {408, 425, 429} or status >= 500:
        raise AgentError(
            f"image service {method} {path} returned transient HTTP {status}: {text}",
            kind=ErrorKind.NOT_AVAILABLE,
        )
    if status >= 300:
        raise AgentError(
            f"image service {method} {path} returned HTTP {status}: {text}",
            kind=ErrorKind.INTERNAL,
        )
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise AgentError(
            f"image service {method} {path} returned invalid JSON: {text[:200]}",
            kind=ErrorKind.INTERNAL,
        ) from exc
    if not isinstance(data, dict):
        raise AgentError(
            f"image service {method} {path} returned non-object JSON",
            kind=ErrorKind.INTERNAL,
        )
    return data


async def _service_image(path: str) -> bytes:
    data, status, content_type = await _curl_image_service("GET", path, None)
    if status in {408, 425, 429} or status >= 500:
        raise AgentError(
            f"image service GET {path} returned transient HTTP {status}",
            kind=ErrorKind.NOT_AVAILABLE,
        )
    if status >= 300:
        raise AgentError(
            f"image service GET {path} returned HTTP {status}: {data[:200]!r}",
            kind=ErrorKind.INTERNAL,
        )
    if len(data) > _MAX_IMAGE_BYTES:
        raise AgentError(
            "image service artifact exceeds size limit", kind=ErrorKind.NOT_AVAILABLE
        )
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise AgentError(
            f"image service returned non-PNG data ({content_type})",
            kind=ErrorKind.NOT_AVAILABLE,
        )
    try:
        from PIL import Image

        with Image.open(BytesIO(data)) as image:
            if image.format != "PNG" or image.width <= 0 or image.height <= 0:
                raise ValueError("invalid PNG format or dimensions")
            image.verify()
    except (ImportError, OSError, ValueError) as exc:
        raise AgentError(
            f"image service returned an invalid PNG artifact: {exc}",
            kind=ErrorKind.NOT_AVAILABLE,
        ) from exc
    return data


async def _curl_image_service(
    method: str,
    path: str,
    body: bytes | None,
    *,
    idempotency_key: str | None = None,
) -> tuple[bytes, int, str]:
    arguments = [
        "/usr/bin/curl",
        "--silent",
        "--show-error",
        "--proxy",
        "",
        "--noproxy",
        "*",
        "--proto",
        "=http",
        "--proto-redir",
        "=http",
        "--max-redirs",
        "0",
        "--request",
        method,
        "--max-time",
        "60",
        "--write-out",
        "\n%{http_code}\n%{content_type}",
    ]
    if body is not None:
        arguments.extend(
            ["--header", "Content-Type: application/json", "--data-binary", "@-"]
        )
    if idempotency_key is not None:
        arguments.extend(["--header", f"Idempotency-Key: {idempotency_key}"])
    arguments.append("http://10.0.0.46:8830" + path)
    process = await asyncio.create_subprocess_exec(
        *arguments,
        stdin=subprocess.PIPE if body is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    output, error = await process.communicate(body)
    if process.returncode != 0:
        detail = error.decode("utf-8", errors="replace").strip()
        raise AgentError(
            f"image service {method} {path} transport failed: {detail}",
            kind=ErrorKind.NOT_AVAILABLE,
        )
    try:
        response, status_text, content_type = output.rsplit(b"\n", 2)
        return (
            response,
            int(status_text),
            content_type.decode("utf-8", errors="replace"),
        )
    except (ValueError, TypeError) as exc:
        raise AgentError(
            f"image service {method} {path} returned malformed transport metadata",
            kind=ErrorKind.INTERNAL,
        ) from exc


def _write_service_image(
    image_data: bytes, output_path: Path, transparent: bool
) -> None:
    """Write the PNG bytes from the service to output_path.

    The service always returns PNG. If the caller asked for JPEG, convert via
    Pillow. Transparent output must stay PNG.
    """
    output_path = output_path.expanduser().absolute()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_components(output_path.parent)
    directory = os.open(
        output_path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    )
    try:
        existing = os.stat(output_path.name, dir_fd=directory, follow_symlinks=False)
    except FileNotFoundError:
        existing = None
    if existing is not None and (
        not stat.S_ISREG(existing.st_mode) or existing.st_uid != os.geteuid()
    ):
        os.close(directory)
        raise PermissionError(f"image output target is unsafe: {output_path}")
    temporary_name = f".{output_path.name}.{uuid.uuid4()}"
    descriptor = os.open(
        temporary_name,
        os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
        dir_fd=directory,
    )
    validation_descriptor = os.dup(descriptor)
    suffix = output_path.suffix.lower()
    try:
        with os.fdopen(descriptor, "wb") as stream:
            if suffix in {".jpg", ".jpeg"}:
                if transparent:
                    raise AgentError(
                        "transparent image output must be PNG, not JPEG",
                        kind=ErrorKind.INVALID_REQUEST,
                    )
                try:
                    from PIL import Image
                except ImportError as exc:
                    raise AgentError(
                        "Pillow is required to save image service PNG output as JPEG",
                        kind=ErrorKind.NOT_AVAILABLE,
                    ) from exc
                with Image.open(BytesIO(image_data)) as image:
                    image.convert("RGB").save(stream, "JPEG", quality=92)
            else:
                stream.write(image_data)
            stream.flush()
            os.fsync(stream.fileno())
        _validate_written_image(validation_descriptor, suffix)
        os.replace(
            temporary_name,
            output_path.name,
            src_dir_fd=directory,
            dst_dir_fd=directory,
        )
        os.fsync(directory)
    finally:
        try:
            os.close(validation_descriptor)
        except OSError:
            pass
        try:
            os.unlink(temporary_name, dir_fd=directory)
        except FileNotFoundError:
            pass
        os.close(directory)


def _validate_written_image(descriptor: int, suffix: str) -> None:
    from PIL import Image

    details = os.fstat(descriptor)
    if (
        not stat.S_ISREG(details.st_mode)
        or details.st_uid != os.geteuid()
        or stat.S_IMODE(details.st_mode) != 0o600
        or details.st_size <= 0
        or details.st_size > _MAX_IMAGE_BYTES
    ):
        raise ValueError("generated image temporary file is unsafe")
    os.lseek(descriptor, 0, os.SEEK_SET)
    with os.fdopen(descriptor, "rb") as stream, Image.open(stream) as image:
        expected = "JPEG" if suffix in {".jpg", ".jpeg"} else "PNG"
        if image.format != expected or image.width <= 0 or image.height <= 0:
            raise ValueError(f"generated image has invalid {expected} metadata")
        image.verify()


def _status_from_service(data: dict[str, Any], requested_job_id: str) -> ImageJobStatus:
    job_id = str(data.get("id") or requested_job_id).strip()
    status = str(data.get("status", "")).strip().lower()
    if job_id != requested_job_id:
        raise AgentError(
            f"image service returned mismatched job id {job_id!r} for {requested_job_id!r}",
            kind=ErrorKind.INTERNAL,
        )
    allowed = {"queued", "running", "done", "failed", "cancelled", "canceled"}
    if status not in allowed:
        raise AgentError(
            f"image service job {job_id} returned unknown status {status!r}",
            kind=ErrorKind.INTERNAL,
        )
    history = data.get("attempt_history", [])
    if not isinstance(history, list) or not all(
        isinstance(item, dict) for item in history
    ):
        raise AgentError(
            f"image service job {job_id} returned invalid attempt provenance",
            kind=ErrorKind.INTERNAL,
        )
    provider = str(data.get("provider") or "codex")
    return ImageJobStatus(
        job_id=job_id,
        status=status,
        ready=status == "done",
        model_used=_CODEX_MODEL,
        provider=provider,
        attempts=int(data.get("attempts") or 0),
        error=str(data.get("error") or ""),
        prompt_version=int(data.get("prompt_version") or 0),
        attempt_history=tuple(dict(item) for item in history),
        created_at=str(data.get("created_at") or ""),
        updated_at=str(data.get("updated_at") or ""),
        provenance=dict(data),
    )


async def get_image_job(job_id: str, *, config: Config | None = None) -> ImageJobStatus:
    _validate_legacy_image_config(config)
    normalized = job_id.strip()
    if not normalized or "/" in normalized:
        raise AgentError(
            "a valid image job id is required", kind=ErrorKind.INVALID_REQUEST
        )
    data = await _service_json("GET", f"/jobs/{normalized}")
    return _status_from_service(data, normalized)


async def download_image_job(
    job_id: str,
    output: str | Path,
    *,
    transparent: bool = False,
    config: Config | None = None,
) -> ImageResult:
    _validate_legacy_image_config(config)
    status = await get_image_job(job_id)
    if status.status in {"failed", "cancelled", "canceled"}:
        raise _terminal_job_error(status)
    if status.status != "done":
        raise AgentError(
            f"image service job {job_id} is {status.status}, not done",
            kind=ErrorKind.INVALID_REQUEST,
            attempts=[{"job_id": job_id, "status": status.status, "recoverable": True}],
        )
    data = await _service_image(f"/jobs/{status.job_id}/image")
    output_path = Path(output).expanduser().absolute()
    _write_service_image(data, output_path, transparent)
    width, height = struct.unpack(">II", data[16:24])
    return ImageResult(
        path=output_path,
        model_used=_CODEX_MODEL,
        conversation_id=uuid.uuid4(),
        prompt="",
        width=width,
        height=height,
        job_id=status.job_id,
        status=status.status,
        ready=True,
        provider=status.provider,
        provenance=status.provenance,
    )


def _terminal_job_error(status: ImageJobStatus) -> AgentError:
    return AgentError(
        f"image service job {status.job_id} ended with status {status.status}: {status.error}",
        kind=ErrorKind.INTERNAL,
        attempts=[
            {
                "job_id": status.job_id,
                "status": status.status,
                "recoverable": False,
            }
        ],
    )


def _submission_error(status: int, data: dict[str, Any], key: str) -> AgentError:
    original_job_id = str(data.get("id") or "")
    code = str(data.get("code") or "")
    message = str(data.get("error") or f"image service returned HTTP {status}")
    metadata = {
        "idempotency_key": key,
        "job_id": original_job_id,
        "status": status,
        "code": code,
        "recoverable": False,
    }
    kind = ErrorKind.INVALID_REQUEST if status in {409, 410} else ErrorKind.INTERNAL
    return AgentError(message, kind=kind, attempts=[metadata])


def _terminal_submission_error(status: int, response: bytes, key: str) -> AgentError:
    """Classify terminal identity failures even when an old service returns text."""
    data: dict[str, Any] = {}
    try:
        decoded = json.loads(response.decode("utf-8", errors="replace"))
        if isinstance(decoded, dict):
            data = decoded
    except ValueError:
        pass
    label = "conflict" if status == 409 else "expired"
    data.setdefault(
        "error", f"image submission idempotency key {label} (HTTP {status})"
    )
    data.setdefault("code", f"idempotency_{label}")
    return _submission_error(status, data, key)


def _parse_image_submission(
    response: bytes, status: int, idempotency_key: str
) -> ImageSubmission:
    if status in {409, 410}:
        raise _terminal_submission_error(status, response, idempotency_key)
    try:
        data = json.loads(response.decode("utf-8", errors="replace"))
    except ValueError as exc:
        raise AgentError(
            "image service POST /jobs returned invalid JSON",
            kind=ErrorKind.INTERNAL,
            attempts=[{"idempotency_key": idempotency_key, "recoverable": True}],
        ) from exc
    if not isinstance(data, dict):
        raise AgentError(
            "image service POST /jobs returned non-object JSON",
            kind=ErrorKind.INTERNAL,
            attempts=[{"idempotency_key": idempotency_key, "recoverable": True}],
        )
    if status != 202:
        raise _submission_error(status, data, idempotency_key)
    job_id = str(data.get("id") or "").strip()
    returned_key = str(data.get("idempotency_key") or "").strip()
    if not job_id or returned_key != idempotency_key:
        raise AgentError(
            "image service returned invalid durable submission identity",
            kind=ErrorKind.INTERNAL,
            attempts=[{"idempotency_key": idempotency_key, "recoverable": True}],
        )
    return ImageSubmission(
        job_id=job_id,
        idempotency_key=idempotency_key,
        replayed=bool(data.get("replayed")),
    )


async def _post_image_job(body: bytes, idempotency_key: str) -> ImageSubmission:
    last_error: AgentError | None = None
    for _ in range(_SUBMISSION_ATTEMPTS):
        try:
            response, status, _ = await _curl_image_service(
                "POST", "/jobs", body, idempotency_key=idempotency_key
            )
        except AgentError as exc:
            if not _is_transient_image_error(exc):
                raise
            last_error = exc
            continue
        if status in {408, 425, 429} or status >= 500:
            last_error = AgentError(
                f"image service submission returned transient HTTP {status}",
                kind=ErrorKind.NOT_AVAILABLE,
            )
            continue
        return _parse_image_submission(response, status, idempotency_key)
    detail = str(last_error) if last_error is not None else "transport failed"
    raise AgentError(
        f"image service submission remains recoverable after transport failure: {detail}",
        kind=ErrorKind.NOT_AVAILABLE,
        attempts=[{"idempotency_key": idempotency_key, "recoverable": True}],
    )


async def _post_image_job_until_accepted(
    body: bytes, idempotency_key: str
) -> ImageSubmission:
    while True:
        try:
            return await _post_image_job(body, idempotency_key)
        except AgentError as exc:
            if not _is_transient_image_error(exc):
                raise
        await _wait_for_poll(2.0)


def _is_transient_image_error(error: AgentError) -> bool:
    return error.kind == ErrorKind.NOT_AVAILABLE


async def submit_image_job(
    prompt: str,
    *,
    width: int,
    height: int,
    image: str | Path | Sequence[str | Path] | None = None,
    transparent: bool = False,
    idempotency_key: str,
    config: Config | None = None,
) -> ImageSubmission:
    _validate_legacy_image_config(config)
    _validate_image_route(prompt, width, height, None, None, None)
    normalized_key = idempotency_key.strip()
    if not normalized_key:
        raise AgentError(
            "idempotency_key is required for direct image submission",
            kind=ErrorKind.INVALID_REQUEST,
        )
    input_paths: Path | list[Path] | None
    if isinstance(image, Sequence) and not isinstance(image, (str, Path)):
        input_paths = [Path(path) for path in image]
    elif image is None:
        input_paths = None
    else:
        input_paths = Path(image)
    payload: dict[str, Any] = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "transparent": transparent,
    }
    sources = _service_input_images(input_paths)
    if sources:
        payload["source_images"] = sources
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return await _post_image_job(body, normalized_key)


async def resume_image_job(
    job_id: str,
    output: str | Path,
    *,
    timeout: float | None = None,
    transparent: bool = False,
    config: Config | None = None,
) -> ImageResult:
    _validate_legacy_image_config(config)
    return await _wait_for_image_job(
        job_id,
        output,
        transparent=transparent,
        timeout=timeout,
    )


def _pending_image_result(status: ImageJobStatus, output: str | Path) -> ImageResult:
    return ImageResult(
        path=Path(output),
        model_used=_CODEX_MODEL,
        conversation_id=uuid.uuid4(),
        prompt="",
        width=0,
        height=0,
        job_id=status.job_id,
        status=status.status,
        ready=False,
        provider=status.provider,
        provenance=status.provenance,
    )


def _pending_operation_result(state_path: Path, state: dict[str, Any]) -> ImageResult:
    job_id = str(state["job_id"])
    status = "accepted" if job_id else "submitting"
    return ImageResult(
        path=Path(str(state["output_path"])),
        model_used=_CODEX_MODEL,
        conversation_id=uuid.uuid4(),
        prompt="",
        width=0,
        height=0,
        job_id=job_id,
        status=status,
        ready=False,
        provider="codex",
        provenance={
            "operation_id": str(state["operation_id"]),
            "operation_state": str(state_path),
            "recoverable": True,
        },
        idempotency_key=str(state["idempotency_key"]),
    )


def _consume_operation_task(
    task: asyncio.Task[tuple[ImageResult, bool, dict[str, Any]]],
) -> None:
    if task.cancelled():
        return
    task.exception()


async def _finish_image_operation(
    state_path: Path,
    body: bytes,
    output: str | Path | None,
    transparent: bool,
    idempotency_key: str | None,
) -> tuple[ImageResult, bool, dict[str, Any]]:
    async with _operation_lock(state_path):
        _, state = _prepare_operation(
            body, output, transparent, idempotency_key, state_path
        )
        replayed = False
        if not state["job_id"]:
            submission = await _post_image_job_until_accepted(
                str(state["request_body"]).encode("utf-8"),
                str(state["idempotency_key"]),
            )
            state["job_id"] = submission.job_id
            replayed = submission.replayed
            _atomic_write_operation(state_path, state)
        result = await _wait_for_image_job(
            str(state["job_id"]),
            Path(str(state["output_path"])),
            transparent=bool(state["transparent"]),
        )
        return result, replayed, state


async def _await_image_operation(
    task: asyncio.Task[tuple[ImageResult, bool, dict[str, Any]]],
    timeout: float | None,
    state_path: Path,
    initial_state: dict[str, Any],
) -> tuple[ImageResult, bool, dict[str, Any]]:
    if timeout is None:
        return await task
    try:
        return await asyncio.wait_for(asyncio.shield(task), max(timeout, 0.0))
    except asyncio.TimeoutError:
        task.add_done_callback(_consume_operation_task)
        try:
            state = _read_operation(state_path)
        except (FileNotFoundError, PermissionError, ValueError):
            state = initial_state
        return _pending_operation_result(state_path, state), False, state


async def _wait_for_image_job(
    job_id: str,
    output: str | Path,
    *,
    transparent: bool = False,
    timeout: float | None = None,
) -> ImageResult:
    loop = asyncio.get_running_loop()
    deadline = None if timeout is None else loop.time() + max(timeout, 0.0)
    last: ImageJobStatus | None = None
    while True:
        if deadline is not None and loop.time() >= deadline and last is not None:
            return _pending_image_result(last, output)
        try:
            status = await get_image_job(job_id)
        except AgentError as exc:
            if not _is_transient_image_error(exc):
                raise
            if deadline is not None and loop.time() >= deadline:
                unknown = ImageJobStatus(
                    job_id=job_id,
                    status="unknown",
                    ready=False,
                    model_used=_CODEX_MODEL,
                    provider="codex",
                )
                return _pending_image_result(unknown, output)
            await _wait_for_poll(_poll_delay(deadline))
            continue
        last = status
        if status.status == "done":
            try:
                return await download_image_job(job_id, output, transparent=transparent)
            except AgentError as exc:
                if not _is_transient_image_error(exc):
                    raise
        elif status.status in {"failed", "cancelled", "canceled"}:
            raise _terminal_job_error(status)
        await _wait_for_poll(_poll_delay(deadline))


def _poll_delay(deadline: float | None) -> float:
    if deadline is None:
        return 2.0
    return min(2.0, max(deadline - asyncio.get_running_loop().time(), 0.0))


# ##################################################################
# generate one image via the mac mini image_generation_service.
async def _wait_for_poll(seconds: float) -> None:
    event = asyncio.Event()
    try:
        await asyncio.wait_for(event.wait(), timeout=seconds)
    except asyncio.TimeoutError:
        return


# ##################################################################
# generate image
#
# There is exactly one image generation path: the mac mini
# image_generation_service at the canonical Mac mini origin, which is backed by
# ChatGPT's image_generation tool over the codex responses endpoint. The
# legacy spark (flux), mflux, nano-banana-2, and codex-CLI backends have
# been removed — codex is the only provider.
def _validate_image_route(
    prompt: str,
    width: int,
    height: int,
    provider: str | None,
    model: str | None,
    steps: int | None,
) -> None:
    if provider is not None and provider.strip().lower() != "codex":
        raise AgentError(
            f"image provider {provider!r} is actively disabled — use the Mac mini Codex image service",
            kind=ErrorKind.INVALID_REQUEST,
        )
    if model is not None and model.strip():
        raise AgentError(
            f"image model {model!r} is actively disabled — the Mac mini Codex image service owns model selection",
            kind=ErrorKind.INVALID_REQUEST,
        )
    if steps is not None:
        raise AgentError(
            "image step overrides are actively disabled — the Mac mini Codex image service owns inference settings",
            kind=ErrorKind.INVALID_REQUEST,
        )
    if not prompt.strip() or width <= 0 or height <= 0:
        raise AgentError(
            "image prompt and positive width/height are required",
            kind=ErrorKind.INVALID_REQUEST,
        )


async def generate_image(
    prompt: str,
    *,
    width: int,
    height: int,
    output: str | Path | None = None,
    image: str | Path | list[str | Path] | None = None,
    tier: Tier = Tier.HIGH,
    transparent: bool = False,
    timeout: float | None = None,
    provider: str | None = None,
    model: str | None = None,
    steps: int | None = None,
    config: Config | None = None,
    logger: ConversationLogger | None = None,
    conversation_id: UUID | None = None,
    idempotency_key: str | None = None,
    operation_state: str | Path | None = None,
) -> ImageResult:
    _validate_legacy_image_config(config)
    _validate_image_route(prompt, width, height, provider, model, steps)

    # validate input image(s) if provided. accept a single path or a list.
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
            input_image_path = (
                paths if len(paths) > 1 else (paths[0] if paths else None)
            )
        else:
            single = Path(image)
            if not single.exists():
                raise AgentError(
                    f"Input image not found: {single}",
                    kind=ErrorKind.INVALID_REQUEST,
                )
            input_image_path = single

    conv_id = conversation_id or uuid.uuid4()

    if logger is not None:
        logger.log_event(
            "image_request",
            prompt=prompt,
            width=width,
            height=height,
            model=_CODEX_MODEL.model_id,
            tier=tier.value,
            transparent=transparent,
            provider="codex",
            fallbacks=[],
        )

    try:
        body = _image_request_body(prompt, width, height, input_image_path, transparent)
        operation_id = _operation_identity(
            body, _operation_output_intent(output), idempotency_key
        )
        state_path = _operation_state_path(operation_id, operation_state)
        _, initial_state = _prepare_operation(
            body, output, transparent, idempotency_key, operation_state
        )
        operation = asyncio.create_task(
            _finish_image_operation(
                state_path,
                body,
                output,
                transparent,
                idempotency_key,
            )
        )
        result, submission_replayed, state = await _await_image_operation(
            operation, timeout, state_path, initial_state
        )
        durable_key = str(state["idempotency_key"])
        job_id = result.job_id
        output_path = result.path
        status = result.status
        ready = result.ready
        replayed = submission_replayed
    except AgentError:
        raise
    except Exception as err:
        raise AgentError(
            f"image generation failed: {err}",
            kind=ErrorKind.INTERNAL,
        ) from err

    if logger is not None:
        logger.log_event(
            "image_complete",
            path=str(output_path),
            provider="codex",
            job_id=job_id,
            status=status,
            ready=ready,
        )

    return ImageResult(
        path=output_path,
        model_used=_CODEX_MODEL,
        conversation_id=conv_id,
        prompt=prompt,
        width=width,
        height=height,
        job_id=job_id,
        status=status,
        ready=ready,
        idempotency_key=durable_key,
        replayed=replayed,
        provenance=result.provenance,
    )


def _image_request_body(
    prompt: str,
    width: int,
    height: int,
    input_image: Path | list[Path] | None,
    transparent: bool,
) -> bytes:
    payload: dict[str, Any] = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "transparent": transparent,
    }
    sources = _service_input_images(input_image)
    if sources:
        payload["source_images"] = sources
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


async def resume_image_operation(
    state_path: str | Path,
    *,
    timeout: float | None = None,
    output: str | Path | None = None,
    config: Config | None = None,
) -> ImageResult:
    _validate_legacy_image_config(config)
    selected = Path(state_path).expanduser().absolute()
    initial_state = _read_operation(selected)
    operation = asyncio.create_task(_finish_resumed_image_operation(selected, output))
    result, _, state = await _await_image_operation(
        operation, timeout, selected, initial_state
    )
    result.idempotency_key = str(state["idempotency_key"])
    return result


async def _finish_resumed_image_operation(
    selected: Path,
    output: str | Path | None,
) -> tuple[ImageResult, bool, dict[str, Any]]:
    async with _operation_lock(selected):
        state = _read_operation(selected)
        if state.get("version") != _OPERATION_STATE_VERSION:
            raise AgentError(
                f"unsupported image operation state version in {selected}",
                kind=ErrorKind.INVALID_REQUEST,
            )
        if output is not None:
            requested = str(Path(output).expanduser().absolute())
            if requested != state.get("output_path"):
                raise AgentError(
                    "recovery output cannot mutate immutable operation metadata",
                    kind=ErrorKind.INVALID_REQUEST,
                )
        if not state.get("job_id"):
            submission = await _post_image_job_until_accepted(
                str(state["request_body"]).encode("utf-8"),
                str(state["idempotency_key"]),
            )
            state["job_id"] = submission.job_id
            _atomic_write_operation(selected, state)
        result = await _wait_for_image_job(
            str(state["job_id"]),
            str(state["output_path"]),
            transparent=bool(state["transparent"]),
        )
        return result, False, state
