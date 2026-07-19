from __future__ import annotations

import math

import pytest

from daz_agent_sdk.providers.arbiter import ArbiterProvider
from daz_agent_sdk.types import AgentError


@pytest.mark.asyncio
async def test_embed_returns_one_vector_per_text(arbiter_tunnel_url: str) -> None:
    provider = ArbiterProvider(base_url=arbiter_tunnel_url)
    result = await provider.embed(["hello world", "embedding test", "third string"])
    assert len(result["embeddings"]) == 3
    assert result["dimension"] == 768
    for vec in result["embeddings"]:
        assert len(vec) == 768
        # L2 norm should be approx 1.0 (nomic-embed is normalized)
        norm = math.sqrt(sum(v * v for v in vec))
        assert 0.98 < norm < 1.02, f"vector norm {norm} is not near 1.0"


@pytest.mark.asyncio
async def test_embed_model_used_is_arbiter_embedding(arbiter_tunnel_url: str) -> None:
    provider = ArbiterProvider(base_url=arbiter_tunnel_url)
    result = await provider.embed(["a single text"])
    assert result["model_repository"] == "nomic-ai/nomic-embed-text-v1.5"
    assert result["dimension"] == 768


@pytest.mark.asyncio
async def test_embed_task_prefix_changes_vectors(arbiter_tunnel_url: str) -> None:
    # Same text with different task prefixes should produce different embeddings.
    text = "what is the capital of france"
    provider = ArbiterProvider(base_url=arbiter_tunnel_url)
    doc = await provider.embed([text], task="search_document")
    query = await provider.embed([text], task="search_query")
    doc_vec = doc["embeddings"][0]
    query_vec = query["embeddings"][0]
    # Cosine similarity between them — same text but different prefixes,
    # should be high but not identical.
    dot = sum(a * b for a, b in zip(doc_vec, query_vec))
    assert 0.85 < dot < 0.9999, (
        f"cosine {dot} — either identical (prefix not applied) or unrelated"
    )


@pytest.mark.asyncio
async def test_embed_empty_list_raises() -> None:
    provider = ArbiterProvider()
    with pytest.raises(AgentError):
        await provider.embed([])
