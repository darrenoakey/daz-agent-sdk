from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID


# ##################################################################
# default log base
# fallback directory used when config does not specify a log location
_DEFAULT_LOG_BASE = Path.home() / ".agent-sdk" / "logs"


# ##################################################################
# conversation logger
# writes structured logs for a single conversation into its own directory
# thread-safe: all file writes are protected by a lock
# all io errors are swallowed — logging failures must never crash the caller
class ConversationLogger:

    # ##################################################################
    # init
    # creates the conversation directory and writes initial meta.json
    # log_base defaults to ~/.agent-sdk/logs when not provided
    def __init__(
        self,
        conversation_uuid: UUID,
        *,
        name: str | None = None,
        tier: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        log_base: Path | str | None = None,
    ) -> None:
        self._uuid = conversation_uuid
        self._lock = threading.Lock()
        self._closed = False

        base = Path(log_base) if log_base is not None else _DEFAULT_LOG_BASE
        self._log_dir = base / str(conversation_uuid)

        try:
            self._log_dir.mkdir(parents=True, exist_ok=True)
            meta = {
                "conversation_id": str(conversation_uuid),
                "name": name,
                "tier": tier,
                "provider": provider,
                "model": model,
                "created_at": _now_iso(),
            }
            (self._log_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        except OSError:
            pass

    # ##################################################################
    # log dir property
    # returns the path to this conversation's log directory
    @property
    def log_dir(self) -> Path:
        return self._log_dir

    # ##################################################################
    # log event
    # appends a single json line to events.jsonl
    # timestamp is added automatically — callers supply event_type and kwargs
    def log_event(self, event_type: str, **kwargs: Any) -> None:
        entry: dict[str, Any] = {"ts": _now_iso(), "event": event_type}
        entry.update(kwargs)
        self._write_jsonl(entry)

    # ##################################################################
    # close
    # writes a final conversation_end event and marks the logger closed
    def close(self, **kwargs: Any) -> None:
        if not self._closed:
            self._closed = True
            self.log_event("conversation_end", **kwargs)

    # ##################################################################
    # write jsonl
    # internal: appends a dict as one json line to events.jsonl
    # acquires the lock and silently ignores all io errors
    def _write_jsonl(self, entry: dict[str, Any]) -> None:
        try:
            line = json.dumps(entry, default=str) + "\n"
            with self._lock:
                with open(self._log_dir / "events.jsonl", "a", encoding="utf-8") as fh:
                    fh.write(line)
        except OSError:
            pass


# ##################################################################
# now iso
# returns the current utc time as an iso 8601 string
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
