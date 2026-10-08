"""问答缓存。

缓存键包含问题、检索参数与对话历史签名，避免多轮场景下复用错答案。
语义缓存（相似问题命中）需要额外一次向量比较，且有答错风险，这里先不做，
在 README 的"下一步"里作为已知优化项列出。
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Sequence

import numpy as np

from .prompt import Message


class QueryCache:
    def __init__(self, max_size: int = 512) -> None:
        self.max_size = max(1, max_size)
        self._store: OrderedDict[str, object] = OrderedDict()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def make_key(
        question: str,
        mode: str,
        top_k: int,
        use_rerank: bool,
        history: Sequence[Message] | None = None,
    ) -> str:
        raw = f"{question.strip()}|{QueryCache.make_scope(mode, top_k, use_rerank, history)}"
        return hashlib.blake2b(raw.encode("utf-8"), digest_size=16).hexdigest()

    @staticmethod
    def make_scope(
        mode: str,
        top_k: int,
        use_rerank: bool,
        history: Sequence[Message] | None = None,
    ) -> str:
        """缓存作用域：检索参数 + 对话历史签名。

        精确缓存把它揉进哈希键；语义缓存用它隔离条目 —— 否则多轮场景下
        一个问题的答案会被复用给上下文完全不同的另一个问题。
        """
        history_signature = "|".join(f"{turn.get('role')}:{turn.get('content', '').strip()}" for turn in (history or [])[-4:])
        return f"{mode}|{top_k}|{use_rerank}|{history_signature}"

    def get(self, key: str):
        if key in self._store:
            self.hits += 1
            self._store.move_to_end(key)
            return self._store[key]
        self.misses += 1
        return None

    def set(self, key: str, value) -> None:
        self._store[key] = value
        self._store.move_to_end(key)
        while len(self._store) > self.max_size:
            self._store.popitem(last=False)

    def clear(self) -> None:
        self._store.clear()

    @property
    def size(self) -> int:
        return len(self._store)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return round(self.hits / total, 4) if total else 0.0


class SemanticCache:
    """语义缓存：措辞不同、意思相同的问题命中同一条答案。

    与 QueryCache（精确匹配）互补：前者只认一模一样的字符串，后者比较向量相似度。
    代价是每次查询多一次编码，且阈值取低了会"把不同的问题当成同一个"。默认关闭，
    阈值与误命中率的关系用离线评测量化（eval/run_cache_risk.py），不要拍脑袋设。
    """

    def __init__(self, max_size: int = 256, threshold: float = 0.95) -> None:
        self.max_size = max(1, max_size)
        self.threshold = float(threshold)
        self.hits = 0
        self.misses = 0
        self._entries: OrderedDict[int, dict] = OrderedDict()
        self._next_id = 0

    def lookup(self, scope: str, vector: np.ndarray) -> tuple[object | None, float]:
        """返回 (命中的答案, 最高相似度)；未达阈值时答案为 None。"""
        best_answer: object | None = None
        best_score = 0.0
        best_id: int | None = None
        for key, entry in self._entries.items():
            if entry["scope"] != scope:
                continue
            score = float(np.dot(entry["vector"], vector))
            if score > best_score:
                best_answer, best_score, best_id = entry["answer"], score, key
        if best_answer is not None and best_score >= self.threshold:
            self.hits += 1
            self._entries.move_to_end(best_id)
            return best_answer, best_score
        self.misses += 1
        return None, best_score

    def set(self, scope: str, vector: np.ndarray, answer: object) -> None:
        """写入缓存；同一作用域内已有近重复问题时替换其答案而不是堆副本。"""
        for key, entry in self._entries.items():
            if entry["scope"] == scope and float(np.dot(entry["vector"], vector)) >= self.threshold:
                entry["answer"] = answer
                self._entries.move_to_end(key)
                return
        self._next_id += 1
        self._entries[self._next_id] = {"scope": scope, "vector": vector, "answer": answer}
        while len(self._entries) > self.max_size:
            self._entries.popitem(last=False)

    def clear(self) -> None:
        self._entries.clear()

    @property
    def size(self) -> int:
        return len(self._entries)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return round(self.hits / total, 4) if total else 0.0
