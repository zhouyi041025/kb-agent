"""中文文本处理：免分词器的分词方案 + 归一化。"""

from __future__ import annotations

import re
import unicodedata

CJK_RANGE = "\u4e00-\u9fff\u3400-\u4dbf"
_CJK_RUN_RE = re.compile(f"[{CJK_RANGE}]+")
_WORD_RE = re.compile(r"[a-z0-9][a-z0-9_+#./-]*")
_SENTENCE_RE = re.compile(r"[^。！？；\n]+[。！？；\n]?")


def normalize(text: str) -> str:
    """NFKC 归一化（全角转半角）+ 小写，保证同一段文本的 token 稳定。"""
    return unicodedata.normalize("NFKC", text).lower()


def tokenize(text: str) -> list[str]:
    """英文数字按词切分，中文按字符 bigram 切分。

    不引第三方分词器：bigram 对未登录词更稳，且索引与查询用同一套切分逻辑，
    避免分词器版本差异导致评测结果不可复现。
    """
    normalized = normalize(text)
    tokens: list[str] = []
    for part in re.split(f"([{CJK_RANGE}]+)", normalized):
        if not part:
            continue
        if _CJK_RUN_RE.fullmatch(part):
            if len(part) == 1:
                tokens.append(part)
            else:
                tokens.extend(part[i : i + 2] for i in range(len(part) - 1))
        else:
            tokens.extend(_WORD_RE.findall(part))
    return tokens


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.findall(text) if s.strip()]


def extract_terms(text: str) -> list[str]:
    """提取英文技术词与中文双字词，用于判定答案是否真正落在上下文里。"""
    terms = tokenize(text)
    seen: dict[str, None] = {}
    for term in terms:
        seen.setdefault(term, None)
    return list(seen)
