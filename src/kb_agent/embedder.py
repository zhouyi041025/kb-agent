"""向量化。

默认的 hashing embedder 是确定性的、零外部依赖的：同样的文本永远得到同样的向量，
因此评测结果可以在任何机器上复现，也不会因为模型版本变化而漂移。
代价是它本质上是词法向量（做了哈希技巧的 bag-of-bigrams），不理解同义词。
需要语义检索时配置 KB_EMBEDDING_PROVIDER=openai 切换即可，接口不变。
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from .config import EmbeddingConfig
from .text import tokenize

DEFAULT_HASHING_DIM = 4096


class Embedder(Protocol):
    dim: int
    name: str

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class HashingEmbedder:
    """signed hashing trick + L2 归一化，行为完全确定。"""

    def __init__(self, dim: int | None = None) -> None:
        resolved = DEFAULT_HASHING_DIM if dim is None else int(dim)
        if resolved <= 0:
            raise ValueError("dim 必须为正整数")
        self.dim = resolved
        self.name = f"hashing-{resolved}"

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

    def __init__(self, model: str, base_url: str, api_key: str, dim: int | None = None, timeout: float = 30.0) -> None:
        if not model:
            raise ValueError("使用 openai embedding 时必须设置 KB_EMBEDDING_MODEL")
        import httpx

        self.model = model
        self.name = f"openai-{model}"
        # auto（None）时维度在首次 encode 后才知道（如 embedding-3 是 2048 维）
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


class CachedEmbedder:
    """从离线 fixture（texts + vectors 的 npz）读取向量。

    用途：把"真实语义向量"的评测变成可复现的离线实验。fixture 由
    scripts/make_embedding_fixture.py 生成一次并入库存档，之后任何人
    clone 下来都能复现同一批语义向量下的评测数字。

    fixture 里查不到的文本直接报错，而不是悄悄回退到词法向量 ——
    两种向量混进同一份报告比报错更危险。
    """

    def __init__(self, path: str | Path) -> None:
        data = np.load(path, allow_pickle=False)
        self._vectors = {text: vector for text, vector in zip(data["texts"].tolist(), data["vectors"])}
        self.dim = int(data["vectors"].shape[1])
        self.name = str(data["model"]) if "model" in data else Path(path).stem

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        missing = [text for text in texts if text not in self._vectors]
        if missing:
            raise KeyError(
                f"离线向量 fixture 缺少 {len(missing)} 条文本的向量，"
                f"请用 scripts/make_embedding_fixture.py 重新生成（首个缺失：{missing[0][:40]}...）"
            )
        return np.stack([self._vectors[text] for text in texts]).astype(np.float32)
