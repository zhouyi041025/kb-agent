"""LLM 客户端。

StubLLM 是离线确定性实现：从检索到的上下文里抽取与问题词重叠最高的句子拼成答案。
它让整条链路（检索 → 组装 → 生成 → 评测）在没有 API Key、没有网络的环境下也能跑通，
所以仓库里的评测数字是可复现的。它当然不是"真"生成模型，README 里明确标注了这一点。
换成真实模型只需设置 KB_LLM_PROVIDER=openai，接口与调用方都不用改。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Protocol, Sequence

from .config import LLMConfig
from .prompt import Message
from .schemas import RetrievedChunk
from .text import split_sentences, tokenize


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    model: str = ""


class LLMClient(Protocol):
    name: str

    def complete(
        self,
        messages: Sequence[Message],
        contexts: Sequence[RetrievedChunk],
        question: str,
    ) -> LLMResult: ...


class StubLLM:
    """抽取式基线：不调用任何外部服务，结果只取决于上下文和问题。"""

    name = "stub-extractive"
    MIN_SCORE = 0.12
    MIN_SHARED_TERMS = 2
    MAX_SENTENCES = 3

    def __init__(self, model: str = "stub-extractive") -> None:
        self.name = model

    def _sentence_score(self, sentence: str, question_terms: set[str]) -> float:
        sentence_terms = set(tokenize(sentence))
        if not sentence_terms or not question_terms:
            return 0.0
        shared = sentence_terms & question_terms
        # 只重合一个高频词（比如"怎么"）不足以支撑回答，直接判为无依据
        if len(shared) < self.MIN_SHARED_TERMS:
            return 0.0
        coverage = len(shared) / len(question_terms)
        length_penalty = 1.0 / (1.0 + math.log(1 + len(sentence_terms) / 40))
        return coverage * length_penalty

    def complete(
        self,
        messages: Sequence[Message],
        contexts: Sequence[RetrievedChunk],
        question: str,
    ) -> LLMResult:
        question_terms = set(tokenize(question))
        scored: list[tuple[float, int, str]] = []
        for index, context in enumerate(contexts, start=1):
            for sentence in split_sentences(context.citation_text):
                score = self._sentence_score(sentence, question_terms)
                if score > 0:
                    scored.append((score, index, sentence))
        scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        selected: list[tuple[int, str]] = []
        used_sentences: set[str] = set()
        for score, index, sentence in scored:
            if score < self.MIN_SCORE or sentence in used_sentences:
                continue
            used_sentences.add(sentence)
            selected.append((index, sentence))
            if len(selected) >= self.MAX_SENTENCES:
                break

        prompt_tokens = sum(len(tokenize(message.get("content", ""))) for message in messages)
        if not selected:
            text = "知识库中没有找到相关依据，无法回答该问题。"
            return LLMResult(text=text, prompt_tokens=prompt_tokens, completion_tokens=len(tokenize(text)), model=self.name)

        lines = [f"- {sentence.strip()} [{index}]" for index, sentence in selected]
        text = "根据知识库内容：\n" + "\n".join(lines)
        return LLMResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=len(tokenize(text)),
            model=self.name,
        )


class OpenAICompatLLM:
    """任何 OpenAI 兼容的 /chat/completions 端点。"""

    # 重试间隔上限（秒）：再长就不如直接降级，让用户看到旧回答
    RETRY_BACKOFF_CAP = 8.0

    # 粗略单价（美元 / 1K token），仅用于成本量级估算，实际以账单为准
    PRICE_TABLE = {
        "gpt-4o-mini": (0.00015, 0.0006),
        "deepseek-chat": (0.00014, 0.00028),
        "glm-4-flash": (0.0, 0.0),  # 智谱的免费档，演示够用
    }

    def __init__(self, config: LLMConfig) -> None:
        if not config.api_key:
            raise ValueError("KB_LLM_PROVIDER=openai 时必须设置 KB_LLM_API_KEY")
        import httpx

        self.config = config
        self.name = config.model
        self._client = httpx.Client(
            base_url=config.base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {config.api_key}"},
            timeout=config.timeout_seconds,
        )

    def _estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        prompt_price, completion_price = self.PRICE_TABLE.get(self.config.model, (0.0, 0.0))
        return round(prompt_tokens / 1000 * prompt_price + completion_tokens / 1000 * completion_price, 6)

    def complete(
        self,
        messages: Sequence[Message],
        contexts: Sequence[RetrievedChunk],
        question: str,
    ) -> LLMResult:
        last_error: Exception | None = None
        for attempt in range(max(0, self.config.max_retries) + 1):
            try:
                response = self._client.post(
                    "/chat/completions",
                    json={
                        "model": self.config.model,
                        "messages": list(messages),
                        "temperature": 0.2,
                    },
                )
                response.raise_for_status()
                payload = response.json()
                usage = payload.get("usage", {})
                prompt_tokens = int(usage.get("prompt_tokens", 0))
                completion_tokens = int(usage.get("completion_tokens", 0))
                # 推理模型（GLM-5.x、DeepSeek-R1 这类）会把思维链放在 reasoning_content，
                # 正文放在 content。如果 token 预算被思考耗尽，content 会是空串或 null。
                message = payload["choices"][0]["message"]
                text = (message.get("content") or "").strip()
                if not text:
                    raise RuntimeError("模型只返回了推理过程，没有正文（content 为空）")
                return LLMResult(
                    text=text,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cost_usd=self._estimate_cost(prompt_tokens, completion_tokens),
                    model=self.config.model,
                )
            except Exception as exc:  # 网络抖动、限流、超时都走重试
                last_error = exc
                if attempt < self.config.max_retries:
                    # 指数退避：被限流后立刻重试只会继续被拒，实测 GLM 的 429
                    # 几乎总是第二次才放行，不加等待的重试等于白试。
                    time.sleep(min(0.5 * 2**attempt, self.RETRY_BACKOFF_CAP))
        raise RuntimeError(f"调用 LLM 失败：{last_error}")


def build_llm(config: LLMConfig) -> LLMClient:
    if config.provider == "openai":
        return OpenAICompatLLM(config)
    return StubLLM()
