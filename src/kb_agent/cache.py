"""问答缓存。

缓存键包含问题、检索参数与对话历史签名，避免多轮场景下复用错答案。
语义缓存（相似问题命中）需要额外一次向量比较，且有答错风险，这里先不做，
在 README 的"下一步"里作为已知优化项列出。
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Sequence

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
        history_signature = "|".join(f"{turn.get('role')}:{turn.get('content', '').strip()}" for turn in (history or [])[-4:])
        raw = f"{question.strip()}|{mode}|{top_k}|{use_rerank}|{history_signature}"
        return hashlib.blake2b(raw.encode("utf-8"), digest_size=16).hexdigest()

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
