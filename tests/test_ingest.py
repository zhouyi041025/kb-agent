from kb_agent.ingest import chunk_document, extract_title
from kb_agent.schemas import Document


def make_document(text: str) -> Document:
    return Document(doc_id="d.md", title="测试文档", text=text)


def test_fixed_strategy_splits_long_text():
    chunks = chunk_document(make_document("段落内容。" * 200), strategy="fixed", size=100, overlap=20)
    assert len(chunks) > 1
    assert all(chunk.chunk_id.startswith("d.md#c") for chunk in chunks)


def test_recursive_strategy_keeps_section_titles():
    document = make_document("# 标题\n\n第一段。\n\n## 小节\n\n第二段。")
    chunks = chunk_document(document, strategy="recursive", size=100, overlap=10)
    assert chunks
    assert any("小节" in chunk.title for chunk in chunks)


def test_parent_child_strategy_links_children_to_parents():
    document = make_document("# 标题\n\n" + "内容。" * 100)
    chunks = chunk_document(document, strategy="parent_child", size=100, overlap=10)
    parent_ids = {chunk.chunk_id for chunk in chunks if chunk.parent_id is None}
    children = [chunk for chunk in chunks if chunk.parent_id]
    assert parent_ids
    assert children
    assert all(child.parent_id in parent_ids for child in children)


def test_chunk_ids_are_deterministic():
    document = make_document("# 标题\n\n第一段。\n\n第二段。")
    first = [chunk.chunk_id for chunk in chunk_document(document, strategy="recursive", size=50, overlap=5)]
    second = [chunk.chunk_id for chunk in chunk_document(document, strategy="recursive", size=50, overlap=5)]
    assert first == second


def test_unknown_strategy_rejected():
    import pytest

    with pytest.raises(ValueError):
        chunk_document(make_document("正文"), strategy="不存在的策略")


def test_extract_title_falls_back_to_first_line():
    assert extract_title("普通正文\n第二行", "fallback") == "普通正文"
