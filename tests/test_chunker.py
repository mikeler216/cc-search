from cc_search.chunker import chunk_text


def test_short_text_single_chunk():
    chunks = chunk_text("Hello, how are you?")
    assert chunks == ["Hello, how are you?"]


def test_empty_text():
    chunks = chunk_text("")
    assert chunks == []


def test_whitespace_only():
    chunks = chunk_text("   \n\n  ")
    assert chunks == []


def test_long_text_splits():
    long_text = "word " * 600
    chunks = chunk_text(long_text, max_tokens=400, overlap_tokens=100)
    assert len(chunks) > 1
    for chunk in chunks:
        word_count = len(chunk.split())
        assert word_count <= 420


def test_overlap_exists():
    words = [f"word{i}" for i in range(800)]
    text = " ".join(words)
    chunks = chunk_text(text, max_tokens=400, overlap_tokens=100)
    assert len(chunks) >= 2
    first_words = set(chunks[0].split())
    second_words = set(chunks[1].split())
    overlap = first_words & second_words
    assert len(overlap) > 0


def test_exact_boundary():
    text = "word " * 400
    chunks = chunk_text(text.strip(), max_tokens=400, overlap_tokens=100)
    assert len(chunks) == 1
