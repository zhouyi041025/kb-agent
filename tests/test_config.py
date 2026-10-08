import os

from kb_agent.config import (
    DEFAULT_MIN_CONFIDENCE,
    AppConfig,
    RetrievalConfig,
    load_env_file,
)
from kb_agent.embedder import HashingEmbedder


def test_env_file_is_loaded(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        '# 注释行会被跳过\nKB_LLM_MODEL="glm-4-flash"\n\nKB_TOP_K=7\n',
        encoding="utf-8",
    )
    monkeypatch.delenv("KB_LLM_MODEL", raising=False)
    monkeypatch.delenv("KB_TOP_K", raising=False)

    assert load_env_file(env_file) is True
    assert os.environ["KB_LLM_MODEL"] == "glm-4-flash"  # 引号会被剥掉
    assert os.environ["KB_TOP_K"] == "7"


def test_env_file_does_not_override_shell_variables(tmp_path, monkeypatch):
    """容器和 CI 注入的环境变量优先级更高，.env 不该盖掉它们。"""
    env_file = tmp_path / ".env"
    env_file.write_text("KB_LLM_MODEL=from-file\n", encoding="utf-8")
    monkeypatch.setenv("KB_LLM_MODEL", "from-shell")

    load_env_file(env_file)
    assert os.environ["KB_LLM_MODEL"] == "from-shell"


def test_missing_env_file_is_not_an_error(tmp_path):
    assert load_env_file(tmp_path / "not-there.env") is False


def test_config_defaults_stay_offline(monkeypatch, tmp_path):
    """.env 不存在时默认仍是离线 stub，clone 下来什么都不用配。"""
    monkeypatch.chdir(tmp_path)
    for key in ("KB_LLM_PROVIDER", "KB_LLM_API_KEY"):
        monkeypatch.delenv(key, raising=False)

    config = AppConfig.from_env()
    assert config.llm.provider == "stub"
    assert config.embedding.provider == "hashing"


def test_min_confidence_default_matches_documented_value(monkeypatch, tmp_path):
    """回归：README 与 dataclass 都标注 0.45，from_env 的 fallback 也必须一致。

    此前 from_env 的 fallback 写成 0.35，导致服务端的实际默认阈值与文档不符：
    文档写的是一条误答率更低的曲线，线上跑的却是另一条。
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("KB_MIN_CONFIDENCE", raising=False)

    config = AppConfig.from_env()
    assert config.retrieval.min_confidence == DEFAULT_MIN_CONFIDENCE == 0.45
    assert RetrievalConfig().min_confidence == config.retrieval.min_confidence


def test_embedding_dim_defaults_to_auto(monkeypatch, tmp_path):
    """回归：维度默认 auto，哈希向量回落到 4096，远端模型按实际维度自适应。"""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("KB_EMBEDDING_DIM", raising=False)

    config = AppConfig.from_env()
    assert config.embedding.dim is None
    assert HashingEmbedder(dim=config.embedding.dim).name == "hashing-4096"


def test_embedding_dim_accepts_explicit_value(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("KB_EMBEDDING_DIM", "2048")
    assert AppConfig.from_env().embedding.dim == 2048

    monkeypatch.setenv("KB_EMBEDDING_DIM", "auto")
    assert AppConfig.from_env().embedding.dim is None
