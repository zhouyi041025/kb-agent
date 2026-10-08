"""配置：全部来自环境变量，方便容器化和 CI 覆盖，不引入 YAML 解析依赖。

工作目录下有 `.env` 时会先加载它（已存在的环境变量优先，不会被覆盖），
这样本地开发不用每次手动 export；容器和 CI 直接注入环境变量即可。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def load_env_file(path: str | Path = ".env") -> bool:
    """把 .env 读进 os.environ，返回是否真的加载了。

    只支持 `KEY=VALUE` 和 `# 注释` 两种语法，不引 python-dotenv：
    需求就这么点，少一个依赖。
    """
    file = Path(path)
    if not file.is_file():
        return False
    for raw in file.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = value.strip().strip('"').strip("'")
    return True


def _env_str(key: str, default: str) -> str:
    value = os.getenv(key)
    return default if value is None or value == "" else value


def _env_int(key: str, default: int) -> int:
    try:
        return int(_env_str(key, str(default)))
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    value = _env_str(key, str(default)).strip().lower()
    return value in {"1", "true", "yes", "on"}


def _env_optional_int(key: str, default: int | None) -> int | None:
    """读一个可选整数：空值 / auto / none 都视为"不指定"。"""
    raw = os.getenv(key)
    if raw is None or raw.strip().lower() in {"", "auto", "none", "null"}:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# 拒答阈值的唯一来源：dataclass 默认值与 from_env 的 fallback 必须一致，
# 否则"文档写 0.45、服务实际跑 0.35"这类偏差会静默进入线上（回归测试见 test_config.py）。
DEFAULT_MIN_CONFIDENCE = 0.45


@dataclass
class LLMConfig:
    provider: str = "stub"
    base_url: str = "https://api.deepseek.com/v1"
    api_key: str = ""
    model: str = "deepseek-chat"
    timeout_seconds: float = 30.0
    max_retries: int = 2

    @property
    def is_remote(self) -> bool:
        return self.provider == "openai"


@dataclass
class EmbeddingConfig:
    provider: str = "hashing"
    # 维度决定签名哈希的碰撞率：512 维时碰撞明显，稠密分支质量被拖累，
    # 4096 维下纯向量分支才达到可用水平（见 eval/report.md 的维度消融）。
    # auto（None）时：哈希向量用默认 4096，远端向量模型以实际返回维度为准，
    # 加载索引时也会按向量文件自适应 —— 避免"按文档切了真实向量模型，
    # 却因为默认维度对不上而在每次启动时静默重建索引"。
    dim: int | None = None
    model: str = ""


@dataclass
class RetrievalConfig:
    mode: str = "hybrid"
    top_k: int = 5
    candidate_k: int = 20
    use_rerank: bool = True
    # 检索置信度门控：最高分片段的 idf 加权词覆盖率低于该值时直接拒答，
    # 宁可说"不知道"也不让模型在没有依据的情况下编答案。
    min_confidence: float = DEFAULT_MIN_CONFIDENCE
    # 置信度取前几个片段的词项并集：答案常常需要跨片段拼装，
    # 只看单个片段会把这类问题误判成"没有依据"（见 eval/report.md 的门控消融）。
    confidence_context_window: int = 3


@dataclass
class AppConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    index_dir: str = ".kb_index"
    knowledge_dir: str = "data/knowledge"
    cache_size: int = 512

    @classmethod
    def from_env(cls) -> "AppConfig":
        load_env_file()
        return cls(
            llm=LLMConfig(
                provider=_env_str("KB_LLM_PROVIDER", "stub"),
                base_url=_env_str("KB_LLM_BASE_URL", "https://api.deepseek.com/v1"),
                api_key=_env_str("KB_LLM_API_KEY", ""),
                model=_env_str("KB_LLM_MODEL", "deepseek-chat"),
                timeout_seconds=float(_env_str("KB_LLM_TIMEOUT_SECONDS", "30")),
                max_retries=_env_int("KB_LLM_MAX_RETRIES", 2),
            ),
            embedding=EmbeddingConfig(
                provider=_env_str("KB_EMBEDDING_PROVIDER", "hashing"),
                dim=_env_optional_int("KB_EMBEDDING_DIM", None),
                model=_env_str("KB_EMBEDDING_MODEL", ""),
            ),
            retrieval=RetrievalConfig(
                mode=_env_str("KB_RETRIEVAL_MODE", "hybrid"),
                top_k=_env_int("KB_TOP_K", 5),
                candidate_k=_env_int("KB_CANDIDATE_K", 20),
                use_rerank=_env_bool("KB_USE_RERANK", True),
                min_confidence=float(_env_str("KB_MIN_CONFIDENCE", str(DEFAULT_MIN_CONFIDENCE))),
            ),
            index_dir=_env_str("KB_INDEX_DIR", ".kb_index"),
            knowledge_dir=_env_str("KB_KNOWLEDGE_DIR", "data/knowledge"),
            cache_size=_env_int("KB_CACHE_SIZE", 512),
        )
