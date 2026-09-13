"""向量化。

默认的 hashing embedder 是确定性的、零外部依赖的：同样的文本永远得到同样的向量，
因此评测结果可以在任何机器上复现，也不会因为模型版本变化而漂移。
代价是它本质上是词法向量（做了哈希技巧的 bag-of-bigrams），不理解同义词。
需要语义检索时配置 KB_EMBEDDING_PROVIDER=openai 切换即可，接口不变。
"""

from __future__ import annotations

import hashlib
from typing import Protocol, Sequence

import numpy as np

from .config import EmbeddingConfig
from .text import tokenize


class Embedder(Protocol):
    dim: int
    name: str

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class HashingEmbedder:
    """signed hashing trick + L2 归一化，行为完全确定。"""

    def __init__(self, dim: int = 512) -> None:
        if dim <= 0:
            raise ValueError("dim 必须为正整数")
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _token_index(self, token: str) -> tuple[int, float]:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "little") % self.dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        return index, sign

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokenize(text):
                index, sign = self._token_index(token)
                vectors[row, index] += sign
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms


class OpenAICompatEmbedder:
    """任何 OpenAI 兼容的 /embeddings 端点（OpenAI、Qwen、GLM、BGE 自建服务等）。"""

    def __init__(self, model: str, base_url: str, api_key: str, dim: int = 1024, timeout: float = 30.0) -> None:
        if not model:
            raise ValueError("使用 openai embedding 时必须设置 KB_EMBEDDING_MODEL")
        import httpx

        self.model = model
        self.name = f"openai-{model}"
        self.dim = dim
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
            timeout=timeout,
        )

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        response = self._client.post("/embeddings", json={"model": self.model, "input": list(texts)})
        response.raise_for_status()
        data = response.json()["data"]
        vectors = np.array([item["embedding"] for item in data], dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.dim = vectors.shape[1]
        return vectors / norms


def build_embedder(config: EmbeddingConfig, *, base_url: str = "", api_key: str = "") -> Embedder:
    if config.provider == "openai":
        return OpenAICompatEmbedder(
            model=config.model,
            base_url=base_url or "https://api.openai.com/v1",
            api_key=api_key,
            dim=config.dim,
        )
    return HashingEmbedder(dim=config.dim)
