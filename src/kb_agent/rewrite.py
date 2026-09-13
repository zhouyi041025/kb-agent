"""查询改写：多轮对话里的指代补全。"""

from __future__ import annotations

import re
from typing import Protocol, Sequence

from .prompt import Message

ANAPHORA = ("它", "他", "这个", "那个", "上述", "该", "此", "这", "那", "上面", "前面", "刚才")
_WHITESPACE_RE = re.compile(r"\s+")


class QueryRewriter(Protocol):
    name: str

    def rewrite(self, question: str, history: Sequence[Message]) -> str: ...


class IdentityRewriter:
    name = "identity"

    def rewrite(self, question: str, history: Sequence[Message]) -> str:
        return question


class HeuristicRewriter:
    """没有历史或没有指代时原样返回；否则把上一轮用户问题拼进来补全语境。

    规则简单但可解释、零延迟。它解决的是最常见的失败模式：
    "它的默认参数是多少？" 单独去检索必然召不回正确文档。
    """

    name = "heuristic"

    def __init__(self, short_question_threshold: int = 15) -> None:
        self.short_question_threshold = short_question_threshold

    def rewrite(self, question: str, history: Sequence[Message]) -> str:
        previous = self._last_user_message(history)
        if not previous:
            return question
        stripped = _WHITESPACE_RE.sub("", question)
        needs_context = len(stripped) <= self.short_question_threshold or any(token in question for token in ANAPHORA)
        if not needs_context:
            return question
        return f"{previous} {question}".strip()

    @staticmethod
    def _last_user_message(history: Sequence[Message]) -> str:
        for turn in reversed(list(history or [])):
            if turn.get("role") == "user":
                return turn.get("content", "").strip()
        return ""


class LLMRewriter:
    """用模型改写，效果更好但增加一次调用（延迟与成本都上升）。"""

    name = "llm"

    def __init__(self, llm, fallback: QueryRewriter | None = None) -> None:
        self.llm = llm
        self.fallback = fallback or HeuristicRewriter()

    def rewrite(self, question: str, history: Sequence[Message]) -> str:
        if not history:
            return question
        from .prompt import REWRITE_SYSTEM_PROMPT

        history_text = "\n".join(f"{turn.get('role')}: {turn.get('content')}" for turn in history[-4:])
        messages = [
            {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": f"对话历史：\n{history_text}\n\n当前问题：{question}"},
        ]
        try:
            result = self.llm.complete(messages, contexts=[], question=question)
        except Exception:
            return self.fallback.rewrite(question, history)
        rewritten = result.text.strip().splitlines()[0].strip() if result.text.strip() else ""
        return rewritten or self.fallback.rewrite(question, history)
