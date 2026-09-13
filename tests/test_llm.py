import pytest

from kb_agent.config import LLMConfig
from kb_agent.llm import OpenAICompatLLM, StubLLM
from kb_agent.schemas import Chunk, RetrievedChunk


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._payload


class FakeClient:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.requests: list[dict] = []
        self.failures = 0

    def post(self, path: str, json: dict) -> FakeResponse:  # noqa: A002 - 对齐 httpx 签名
        self.requests.append({"path": path, "json": json})
        if len(self.requests) <= self.failures:
            raise RuntimeError('429 {"error":{"code":"1305","message":"该模型当前访问量过大"}}')
        return FakeResponse(self.payload)


def make_llm(payload: dict) -> OpenAICompatLLM:
    config = LLMConfig(provider="openai", api_key="test-key", model="glm-5.3-flash", max_retries=0)
    llm = OpenAICompatLLM(config)
    llm._client = FakeClient(payload)  # noqa: SLF001 - 单测里替换传输层
    return llm


def test_openai_client_returns_message_content():
    llm = make_llm(
        {
            "choices": [{"message": {"role": "assistant", "content": "答案 [1]"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        }
    )
    result = llm.complete([{"role": "user", "content": "问题"}], contexts=[], question="问题")

    assert result.text == "答案 [1]"
    assert (result.prompt_tokens, result.completion_tokens) == (10, 5)
    assert llm._client.requests[0]["path"] == "/chat/completions"


def test_openai_client_rejects_reasoning_only_response():
    """推理模型可能把预算全用在思考上、正文留空；这种情况要报错走降级，而不是崩在 .strip()。"""
    llm = make_llm(
        {"choices": [{"message": {"role": "assistant", "content": None, "reasoning_content": "想了很久…"}}]}
    )
    with pytest.raises(RuntimeError, match="调用 LLM 失败"):
        llm.complete([{"role": "user", "content": "问题"}], contexts=[], question="问题")


def test_openai_client_requires_api_key():
    with pytest.raises(ValueError, match="KB_LLM_API_KEY"):
        OpenAICompatLLM(LLMConfig(provider="openai", api_key=""))


def make_retrying_llm(payload: dict, failures: int, max_retries: int) -> tuple[OpenAICompatLLM, FakeClient]:
    config = LLMConfig(provider="openai", api_key="test-key", model="glm-4.7-flash", max_retries=max_retries)
    llm = OpenAICompatLLM(config)
    client = FakeClient(payload)
    client.failures = failures
    llm._client = client  # noqa: SLF001 - 单测里替换传输层
    return llm, client


def test_retries_with_exponential_backoff(monkeypatch):
    """被限流后必须等待再重试；立刻重试只会继续被拒。"""
    slept: list[float] = []
    monkeypatch.setattr("kb_agent.llm.time.sleep", lambda seconds: slept.append(seconds))

    llm, client = make_retrying_llm({"choices": [{"message": {"content": "答案"}}]}, failures=2, max_retries=2)
    result = llm.complete([{"role": "user", "content": "问题"}], contexts=[], question="问题")

    assert result.text == "答案"
    assert len(client.requests) == 3
    assert slept == [0.5, 1.0]  # 指数退避


def test_gives_up_after_max_retries(monkeypatch):
    monkeypatch.setattr("kb_agent.llm.time.sleep", lambda seconds: None)

    llm, client = make_retrying_llm({"choices": [{"message": {"content": "答案"}}]}, failures=99, max_retries=2)
    with pytest.raises(RuntimeError, match="调用 LLM 失败"):
        llm.complete([{"role": "user", "content": "问题"}], contexts=[], question="问题")

    assert len(client.requests) == 3  # 首次 + 2 次重试


def test_backoff_is_capped(monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr("kb_agent.llm.time.sleep", lambda seconds: slept.append(seconds))

    llm, _ = make_retrying_llm({"choices": [{"message": {"content": "答案"}}]}, failures=99, max_retries=5)
    with pytest.raises(RuntimeError):
        llm.complete([{"role": "user", "content": "问题"}], contexts=[], question="问题")

    assert slept == [0.5, 1.0, 2.0, 4.0, 8.0]  # 到上限就不再增长


def make_context(text: str) -> RetrievedChunk:
    chunk = Chunk("d1#c0", "d1.md", "混合检索", text)
    return RetrievedChunk(chunk=chunk, score=1.0, retriever="bm25", rank=1)


def test_stub_answers_from_the_matching_sentence():
    """stub 挑句子的依据是 bigram 重合度，会选重合最多的那句，并带上引用编号。"""
    stub = StubLLM()
    contexts = [make_context("固定长度切分按字符数硬切。混合检索用 BM25 与向量两路召回，再用 RRF 融合。")]

    result = stub.complete([{"role": "user", "content": "混合检索怎么融合"}], contexts=contexts, question="混合检索怎么融合")

    assert "混合检索" in result.text
    assert "[1]" in result.text


def test_stub_refuses_when_nothing_matches():
    stub = StubLLM()
    contexts = [make_context("固定长度切分按字符数硬切，保留重叠窗口。")]

    result = stub.complete([{"role": "user", "content": "今天天气怎么样"}], contexts=contexts, question="今天天气怎么样")

    assert "没有找到相关依据" in result.text
