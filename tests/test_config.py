import os

from kb_agent.config import AppConfig, load_env_file


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
