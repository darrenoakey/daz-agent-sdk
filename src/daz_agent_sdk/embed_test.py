from __future__ import annotations

import math
import socket
from urllib.parse import urlparse

import pytest
import pytest_asyncio  # noqa: F401 — registers asyncio mode

from daz_agent_sdk import agent
from daz_agent_sdk.types import AgentError, Capability, EmbeddingResult


_DEFAULT_URL = "http://10.0.0.254:8400"


def _arbiter_reachable() -> bool:
    parsed = urlparse(_DEFAULT_URL)
    host = parsed.hostname or "10.0.0.254"
    port = parsed.port or 8400
    try:
        s = socket.create_connection((host, port), timeout=2)
        s.close()
        return True
    except OSError:
        return False


ARBITER_RUNNING = _arbiter_reachable()
skip_if_no_arbiter = pytest.mark.skipif(
    not ARBITER_RUNNING,
    reason="Arbiter not reachable at 10.0.0.254:8400",
)


@skip_if_no_arbiter
@pytest.mark.asyncio
async def test_embed_returns_one_vector_per_text() -> None:
    result = await agent.embed(["hello world", "embedding test", "third string"])
    assert isinstance(result, EmbeddingResult)
    assert len(result.embeddings) == 3
    assert result.dimension == 768
    for vec in result.embeddings:
        assert len(vec) == 768
        # L2 norm should be approx 1.0 (nomic-embed is normalized)
        norm = math.sqrt(sum(v * v for v in vec))
        assert 0.98 < norm < 1.02, f"vector norm {norm} is not near 1.0"


@skip_if_no_arbiter
@pytest.mark.asyncio
async def test_embed_model_used_is_arbiter_embedding() -> None:
    result = await agent.embed(["a single text"])
    assert result.model_used.provider == "arbiter"
    assert result.model_used.model_id == "embed-text"
    assert Capability.EMBEDDING in result.model_used.capabilities


@skip_if_no_arbiter
@pytest.mark.asyncio
async def test_embed_task_prefix_changes_vectors() -> None:
    # Same text with different task prefixes should produce different embeddings.
    text = "what is the capital of france"
    doc = await agent.embed([text], task="search_document")
    query = await agent.embed([text], task="search_query")
    doc_vec = doc.embeddings[0]
    query_vec = query.embeddings[0]
    # Cosine similarity between them — same text but different prefixes,
    # should be high but not identical.
    dot = sum(a * b for a, b in zip(doc_vec, query_vec))
    assert 0.85 < dot < 0.9999, (
        f"cosine {dot} — either identical (prefix not applied) or unrelated"
    )


@pytest.mark.asyncio
async def test_embed_empty_list_raises() -> None:
    with pytest.raises(AgentError):
        await agent.embed([])
