"""Prompt 组装。

约束模型只用给定上下文回答、必须带引用编号、证据不足时明确说不知道：
这三条直接决定了线上幻觉率和"答非所问"的比例，是 RAG 里性价比最高的 prompt 设计。
"""

from __future__ import annotations

from typing import Sequence

from .schemas import RetrievedChunk

Message = dict[str, str]

SYSTEM_PROMPT = """你是企业内部知识库助手。请严格遵守：
1. 只使用 <context> 中给出的信息回答，不要引入外部知识。
2. 每个结论后面用方括号加数字标注来源，例如 [1]、[2]，数字对应 <context id="1">、<context id="2">。
   方括号里只能有数字，不要写"编号"两个字，也不要写其他文字。
3. 如果上下文不足以回答，直接说明"知识库中没有找到相关依据"，不要猜测。
4. 回答用中文，先给结论再给依据，控制在 200 字以内；不要复述问题本身。
5. 需要列举多条时，每条独占一行，用 "1. "、"2. " 编号，不要把多条挤在同一行。"""

REWRITE_SYSTEM_PROMPT = """你是检索查询改写器。把用户问题改写成一条适合检索的独立查询：
- 补全指代（它、这个、上述等）为上文中的具体对象
- 保留专有名词、缩写、数字
- 只输出改写后的查询本身，不要解释、不要加引号"""


def format_contexts(contexts: Sequence[RetrievedChunk]) -> str:
    blocks = []
    for index, context in enumerate(contexts, start=1):
        blocks.append(
            f'<context id="{index}" source="{context.chunk.title}">\n{context.citation_text.strip()}\n</context>'
        )
    return "\n\n".join(blocks)


def build_messages(question: str, contexts: Sequence[RetrievedChunk], history: Sequence[Message] | None = None) -> list[Message]:
    messages: list[Message] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in (history or [])[-4:]:
        role = turn.get("role", "user")
        if role not in {"user", "assistant"}:
            continue
        messages.append({"role": role, "content": turn.get("content", "")})
    user_content = f"{format_contexts(contexts)}\n\n用户问题：{question}"
    messages.append({"role": "user", "content": user_content})
    return messages
