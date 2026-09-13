from kb_agent.text import normalize, split_sentences, tokenize


def test_cjk_tokenized_as_bigrams():
    assert tokenize("知识库") == ["知识", "识库"]


def test_single_cjk_character_kept():
    assert tokenize("库") == ["库"]


def test_mixed_language_keeps_ascii_words_whole():
    tokens = tokenize("RAG 检索")
    assert "rag" in tokens
    assert "检索" in tokens


def test_normalize_applies_nfkc_and_lowercase():
    assert normalize("ＡＢＣ１２３") == "abc123"


def test_split_sentences_keeps_delimiters():
    assert split_sentences("你好。世界！") == ["你好。", "世界！"]
