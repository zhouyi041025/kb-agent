"""缓存行为：精确匹配（QueryCache）与语义匹配（SemanticCache）。"""

import numpy as np

from kb_agent.cache import QueryCache, SemanticCache


def unit(*values: float) -> np.ndarray:
    vector = np.array(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def test_query_cache_key_includes_history():
    """多轮场景下同一个问题配不同上文，不能复用同一份答案。"""
    key_a = QueryCache.make_key("问题", "hybrid", 5, True, [{"role": "user", "content": "上文A"}])
    key_b = QueryCache.make_key("问题", "hybrid", 5, True, [{"role": "user", "content": "上文B"}])
    assert key_a != key_b


def test_query_cache_evicts_least_recently_used():
    cache = QueryCache(max_size=2)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.get("a")
    cache.set("c", 3)

    assert cache.size == 2
    assert cache.get("b") is None
    assert cache.get("a") == 1
    assert cache.get("c") == 3


def test_semantic_cache_hits_on_close_vector():
    cache = SemanticCache(threshold=0.95)
    cache.set("scope", unit(1, 0), "answer-1")

    hit, score = cache.lookup("scope", unit(1, 0))
    assert hit == "answer-1"
    assert score >= 0.95
    assert cache.hits == 1


def test_semantic_cache_misses_below_threshold():
    cache = SemanticCache(threshold=0.95)
    cache.set("scope", unit(1, 0), "answer-1")

    hit, _score = cache.lookup("scope", unit(0, 1))
    assert hit is None
    assert cache.misses == 1


def test_semantic_cache_isolates_scopes():
    """检索参数或对话历史不同 = 不同作用域，不能互相命中。"""
    cache = SemanticCache(threshold=0.95)
    cache.set("hybrid|5|True|", unit(1, 0), "hybrid 答案")

    hit, _score = cache.lookup("bm25|5|True|", unit(1, 0))
    assert hit is None


def test_semantic_cache_replaces_near_duplicates():
    cache = SemanticCache(threshold=0.9)
    cache.set("scope", unit(1, 0), "旧答案")
    cache.set("scope", unit(1, 0.01), "新答案")

    assert cache.size == 1
    hit, _score = cache.lookup("scope", unit(1, 0))
    assert hit == "新答案"


def test_semantic_cache_evicts_lru():
    cache = SemanticCache(max_size=2, threshold=0.99)
    cache.set("scope", unit(1, 0), "a")
    cache.set("scope", unit(0, 1), "b")
    cache.set("scope", unit(0.7071, 0.7071), "c")

    assert cache.size == 2
