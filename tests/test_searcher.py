import os
import tempfile

from cc_search.searcher import Searcher
from cc_search.db import SearchDB


def _seed_db(db, model):
    texts = [
        ("How do I set up JWT auth middleware?", "user"),
        ("Here is how to configure JWT validation.", "assistant"),
        ("Fix the database migration bug", "user"),
        ("The migration failed because of a missing column.", "assistant"),
        ("How does React state management work?", "user"),
    ]
    chunks = []
    embeddings = []
    for i, (text, role) in enumerate(texts):
        chunks.append({
            "file_path": "/test.jsonl",
            "session_id": "sess-1",
            "project": "Users/test/myproject",
            "role": role,
            "chunk_text": text,
            "turn_index": i,
            "created_at": 1000.0 + i,
        })
        embeddings.append(model.encode(text).tolist())
    db.insert_chunks_batch(chunks, embeddings)
    db.commit()


def test_search_returns_relevant_results(model):
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        searcher = Searcher(db_path=db_path, model_name=None)
        searcher._model = model
        _seed_db(searcher.db, model)
        results = searcher.search("authentication setup")
        assert len(results) > 0
        top_text = results[0]["chunk_text"]
        assert "JWT" in top_text or "auth" in top_text


def test_search_top_k(model):
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        searcher = Searcher(db_path=db_path, model_name=None)
        searcher._model = model
        _seed_db(searcher.db, model)
        results = searcher.search("anything", top_k=2)
        assert len(results) <= 2


def test_search_empty_db(model):
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        searcher = Searcher(db_path=db_path, model_name=None)
        searcher._model = model
        results = searcher.search("hello")
        assert results == []


def test_search_with_resume_command(model):
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        searcher = Searcher(db_path=db_path, model_name=None)
        searcher._model = model
        _seed_db(searcher.db, model)
        results = searcher.search("database migration")
        assert len(results) > 0
        assert results[0]["resume_command"] == "claude --resume sess-1"
