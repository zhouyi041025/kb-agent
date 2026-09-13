"""文档加载与切分。

支持三种切分策略，用于消融对比：
- fixed：按固定字符长度切，最朴素，容易切断语义
- recursive：按标题/段落递归合并到目标长度，保持段落完整
- parent_child：子块用于检索（粒度细、召回准），父块用于生成（上下文全）
"""

from __future__ import annotations

import re
from pathlib import Path

from .schemas import Chunk, Document

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".pdf"}
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - 取决于可选依赖
        raise RuntimeError("解析 PDF 需要安装 pypdf：pip install pypdf") from exc
    reader = PdfReader(str(path))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def read_text_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return _read_pdf(path)
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def extract_title(text: str, fallback: str) -> str:
    match = _HEADING_RE.search(text)
    if match:
        return match.group(2).strip()
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:80]
    return fallback


def load_documents(root: str | Path) -> list[Document]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"知识库目录不存在：{root}")
    documents: list[Document] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        text = read_text_file(path).strip()
        if not text:
            continue
        relative = path.relative_to(root).as_posix()
        documents.append(
            Document(
                doc_id=relative,
                title=extract_title(text, path.stem),
                text=text,
                source=str(path),
            )
        )
    return documents


def _split_sections(text: str) -> list[tuple[str, str]]:
    """按 Markdown 标题切成 (heading, body)，没有标题时整篇算一节。"""
    matches = list(_HEADING_RE.finditer(text))
    if not matches:
        return [("", text)]
    sections: list[tuple[str, str]] = []
    if matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.append(("", preamble))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.end() : end].strip()
        sections.append((match.group(2).strip(), body))
    return sections


def _split_paragraphs(text: str) -> list[str]:
    return [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]


def _hard_split(text: str, size: int, overlap: int) -> list[str]:
    step = max(1, size - overlap)
    return [text[start : start + size] for start in range(0, len(text), step) if text[start : start + size].strip()]


def _merge_paragraphs(paragraphs: list[str], size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    buffer = ""
    for paragraph in paragraphs:
        pieces = [paragraph] if len(paragraph) <= size else _hard_split(paragraph, size, overlap)
        for piece in pieces:
            if not buffer:
                buffer = piece
            elif len(buffer) + len(piece) + 2 <= size:
                buffer = f"{buffer}\n\n{piece}"
            else:
                chunks.append(buffer)
                tail = buffer[-overlap:] if overlap > 0 else ""
                buffer = f"{tail}\n\n{piece}".strip() if tail else piece
    if buffer:
        chunks.append(buffer)
    return chunks


def chunk_document(
    document: Document,
    strategy: str = "recursive",
    size: int = 400,
    overlap: int = 80,
) -> list[Chunk]:
    if strategy == "fixed":
        pieces = _hard_split(document.text, size, overlap)
        return [
            Chunk(
                chunk_id=f"{document.doc_id}#c{index}",
                doc_id=document.doc_id,
                title=document.title,
                text=piece,
                position=index,
                source=document.source,
            )
            for index, piece in enumerate(pieces)
        ]

    if strategy == "recursive":
        chunks: list[Chunk] = []
        for section_index, (heading, body) in enumerate(_split_sections(document.text)):
            for piece in _merge_paragraphs(_split_paragraphs(body), size, overlap):
                position = len(chunks)
                chunks.append(
                    Chunk(
                        chunk_id=f"{document.doc_id}#c{position}",
                        doc_id=document.doc_id,
                        title=f"{document.title} · {heading}" if heading else document.title,
                        text=piece,
                        position=position,
                        source=document.source,
                    )
                )
        return chunks

    if strategy == "parent_child":
        chunks = []
        for section_index, (heading, body) in enumerate(_split_sections(document.text)):
            parent_id = f"{document.doc_id}#p{section_index}"
            parent_text = f"{heading}\n\n{body}".strip() if heading else body
            children = _merge_paragraphs(_split_paragraphs(body), max(1, size // 2), overlap // 2)
            for child in children:
                chunks.append(
                    Chunk(
                        chunk_id=f"{document.doc_id}#c{len(chunks)}",
                        doc_id=document.doc_id,
                        title=f"{document.title} · {heading}" if heading else document.title,
                        text=child,
                        position=len(chunks),
                        parent_id=parent_id,
                        source=document.source,
                    )
                )
            # 父块自身也入索引一份，保证消融对比时两种粒度都可见
            chunks.append(
                Chunk(
                    chunk_id=parent_id,
                    doc_id=document.doc_id,
                    title=f"{document.title} · {heading}" if heading else document.title,
                    text=parent_text,
                    position=len(chunks),
                    source=document.source,
                )
            )
        return chunks

    raise ValueError(f"未知切分策略：{strategy}")


def build_chunks(
    documents: list[Document],
    strategy: str = "recursive",
    size: int = 400,
    overlap: int = 80,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(chunk_document(document, strategy=strategy, size=size, overlap=overlap))
    return chunks
