from __future__ import annotations

import json
import os
import stat
import threading
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from daz_agent_sdk.logging_ import ConversationLogger


# ##################################################################
# helpers
# read all events from a conversation's events.jsonl
def _read_events(log_dir: Path) -> list[dict]:
    path = log_dir / "events.jsonl"
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


# ##################################################################
# meta json creation
# verify the directory and meta.json are created with correct fields
def test_meta_json_created(tmp_path: Path) -> None:
    uid = uuid4()
    logger = ConversationLogger(
        uid,
        name="test-convo",
        tier="high",
        provider="claude",
        model="claude-opus-4-6",
        log_base=tmp_path,
    )
    meta_path = logger.log_dir / "meta.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["conversation_id"] == str(uid)
    assert meta["name"] == "test-convo"
    assert meta["tier"] == "high"
    assert meta["provider"] == "claude"
    assert meta["model"] == "claude-opus-4-6"
    assert "created_at" in meta


# ##################################################################
# meta json with none fields
# verify none values are serialised rather than raising
def test_meta_json_none_fields(tmp_path: Path) -> None:
    uid = uuid4()
    logger = ConversationLogger(uid, log_base=tmp_path)
    meta = json.loads((logger.log_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["name"] is None
    assert meta["tier"] is None
    assert meta["provider"] is None
    assert meta["model"] is None


# ##################################################################
# log dir property
# verify log_dir points to {log_base}/{uuid}/
def test_log_dir_property(tmp_path: Path) -> None:
    uid = uuid4()
    logger = ConversationLogger(uid, log_base=tmp_path)
    assert logger.log_dir == tmp_path / str(uid)
    assert logger.log_dir.is_dir()


# ##################################################################
# log event basic
# verify a single event is appended to events.jsonl with timestamp
def test_log_event_basic(tmp_path: Path) -> None:
    uid = uuid4()
    logger = ConversationLogger(uid, log_base=tmp_path)
    logger.log_event("conversation_start", tier="medium", provider="claude")
    events = _read_events(logger.log_dir)
    assert len(events) == 1
    assert events[0]["event"] == "conversation_start"
    assert events[0]["tier"] == "medium"
    assert events[0]["provider"] == "claude"
    assert "ts" in events[0]


# ##################################################################
# log event multiple
# verify multiple events are appended in order as valid jsonl
def test_log_event_multiple(tmp_path: Path) -> None:
    uid = uuid4()
    logger = ConversationLogger(uid, log_base=tmp_path)
    logger.log_event("conversation_start")
    logger.log_event("user_message", turn=1, content="hello")
    logger.log_event("assistant_response", turn=1, content="hi", tokens=10)
    events = _read_events(logger.log_dir)
    assert len(events) == 3
    assert events[0]["event"] == "conversation_start"
    assert events[1]["event"] == "user_message"
    assert events[1]["turn"] == 1
    assert events[2]["event"] == "assistant_response"
    assert events[2]["tokens"] == 10


# ##################################################################
# all event types
# verify all documented event types can be logged without error
def test_all_documented_event_types(tmp_path: Path) -> None:
    uid = uuid4()
    logger = ConversationLogger(uid, log_base=tmp_path)
    event_types = [
        "conversation_start",
        "user_message",
        "assistant_response",
        "rate_limit",
        "backoff",
        "cascade",
        "image_request",
        "image_complete",
        "structured_output",
        "conversation_end",
        "error",
    ]
    for et in event_types:
        logger.log_event(et)
    events = _read_events(logger.log_dir)
    logged = [e["event"] for e in events]
    for et in event_types:
        assert et in logged


# ##################################################################
# timestamps are iso 8601
# verify ts field can be parsed as a valid utc datetime
def test_timestamps_are_valid_iso(tmp_path: Path) -> None:
    uid = uuid4()
    logger = ConversationLogger(uid, log_base=tmp_path)
    logger.log_event("user_message", content="test")
    events = _read_events(logger.log_dir)
    ts = events[0]["ts"]
    # must parse without raising
    parsed = datetime.fromisoformat(ts)
    # must carry timezone info
    assert parsed.tzinfo is not None


# ##################################################################
# thread safety
# log from many threads concurrently and verify no lines are lost or corrupt
def test_thread_safety(tmp_path: Path) -> None:
    uid = uuid4()
    logger = ConversationLogger(uid, log_base=tmp_path)
    n_threads = 20
    events_per_thread = 10
    barrier = threading.Barrier(n_threads)

    def worker(thread_id: int) -> None:
        barrier.wait()
        for i in range(events_per_thread):
            logger.log_event("user_message", thread=thread_id, seq=i)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    events = _read_events(logger.log_dir)
    assert len(events) == n_threads * events_per_thread
    # every line must be valid json
    for e in events:
        assert "event" in e
        assert "ts" in e


# ##################################################################
# ioerror resilience
# verify that logging on a read-only directory does not raise
def test_ioerror_resilience(tmp_path: Path) -> None:
    uid = uuid4()
    # create logger successfully first
    logger = ConversationLogger(uid, log_base=tmp_path)
    # make the log directory read-only so writes fail
    original_mode = logger.log_dir.stat().st_mode
    try:
        os.chmod(logger.log_dir, stat.S_IRUSR | stat.S_IXUSR)
        # these must not raise even though writes will fail
        logger.log_event("user_message", content="will fail silently")
        logger.close()
    finally:
        os.chmod(logger.log_dir, original_mode)


# ##################################################################
# ioerror on init
# verify that a bad log_base (file instead of dir) does not raise
def test_ioerror_on_bad_log_base(tmp_path: Path) -> None:
    # create a file where log_base would be — init must not raise
    bad_base = tmp_path / "is_a_file"
    bad_base.write_text("blocked", encoding="utf-8")
    uid = uuid4()
    # must not raise
    logger = ConversationLogger(uid, log_base=bad_base)
    # log_event must not raise either
    logger.log_event("conversation_start")


# ##################################################################
# close writes conversation_end
# verify close() appends exactly one conversation_end event
def test_close_writes_conversation_end(tmp_path: Path) -> None:
    uid = uuid4()
    logger = ConversationLogger(uid, log_base=tmp_path)
    logger.log_event("conversation_start")
    logger.log_event("user_message", turn=1)
    logger.close(turns=1, total_tokens=50)
    events = _read_events(logger.log_dir)
    end_events = [e for e in events if e["event"] == "conversation_end"]
    assert len(end_events) == 1
    assert end_events[0]["turns"] == 1
    assert end_events[0]["total_tokens"] == 50


# ##################################################################
# close is idempotent
# calling close() twice must not produce two conversation_end events
def test_close_idempotent(tmp_path: Path) -> None:
    uid = uuid4()
    logger = ConversationLogger(uid, log_base=tmp_path)
    logger.close()
    logger.close()
    events = _read_events(logger.log_dir)
    end_events = [e for e in events if e["event"] == "conversation_end"]
    assert len(end_events) == 1


# ##################################################################
# jsonl format
# every line in events.jsonl must be independently parseable json
def test_events_jsonl_format(tmp_path: Path) -> None:
    uid = uuid4()
    logger = ConversationLogger(uid, log_base=tmp_path)
    logger.log_event("conversation_start")
    logger.log_event("user_message", content="hi there")
    logger.log_event("assistant_response", content="hello", tokens=5)
    logger.close()

    raw_lines = (logger.log_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(raw_lines) == 4  # 3 + close
    for line in raw_lines:
        obj = json.loads(line)
        assert isinstance(obj, dict)
        assert "ts" in obj
        assert "event" in obj


# ##################################################################
# uuid string logging
# verify uuid values in kwargs are serialised as strings not objects
def test_uuid_kwargs_serialise(tmp_path: Path) -> None:
    uid = uuid4()
    turn_id = uuid4()
    logger = ConversationLogger(uid, log_base=tmp_path)
    logger.log_event("assistant_response", turn_id=turn_id)
    events = _read_events(logger.log_dir)
    assert events[0]["turn_id"] == str(turn_id)
