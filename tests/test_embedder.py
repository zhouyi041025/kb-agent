"""向量化器的边界行为，重点是离线 fixture（CachedEmbedder）。"""

import numpy as np
import pytest

from kb_agent.embedder import CachedEmbedder, HashingEmbedder


def make_fixture(tmp_path):
    embedder = HashingEmbedder(dim=64)
    texts = ["混合检索", "切分策略"]
    vectors = embedder.encode(texts)
    path = tmp_path / "fixture.npz"
    np.savez(path, texts=np.array(texts), vectors=vectors, model=np.array("hashing-64-test"))
    return path, texts, vectors


def test_cached_embedder_reads_vectors_from_fixture(tmp_path):
    path, texts, vectors = make_fixture(tmp_path)

    cached = CachedEmbedder(path)
    assert cached.name == "hashing-64-test"
    assert cached.dim == 64
    assert np.allclose(cached.encode(texts), vectors)


def test_cached_embedder_fails_loudly_on_missing_text(tmp_path):
    """fixture 缺向量时必须报错：悄悄回退到别的向量会让报告混入两种向量。"""
    path, _texts, _vectors = make_fixture(tmp_path)

    with pytest.raises(KeyError, match="缺少"):
        CachedEmbedder(path).encode(["fixture 里没有的文本"])


def test_hashing_embedder_defaults_to_4096():
    assert HashingEmbedder().name == "hashing-4096"
