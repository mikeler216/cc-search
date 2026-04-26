import os
import tempfile

from cc_search.db import SearchDB


def test_create_tables():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        with SearchDB(db_path) as db:
            cursor = db._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            tables = [row[0] for row in cursor.fetchall()]
            assert "chunks" in tables
            assert "files" in tables


def test_insert_and_get_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        with SearchDB(db_path) as db:
            db.upsert_file("/some/path.jsonl", last_modified=1000.0, last_indexed=1001.0)
            db.commit()
            row = db.get_file("/some/path.jsonl")
            assert row is not None
            assert row["last_modified"] == 1000.0


def _insert_one(db, embedding=None):
    if embedding is None:
        embedding = [0.1] * 384
    chunk = {
        "file_path": "/path.jsonl",
        "session_id": "abc-123",
        "project": "my-project",
        "role": "user",
        "chunk_text": "hello world",
        "turn_index": 0,
        "created_at": 1000.0,
    }
    db.insert_chunks_batch([chunk], [embedding])
    db.commit()


def test_insert_batch_and_search_vec():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        with SearchDB(db_path) as db:
            embedding = [0.1] * 384
            _insert_one(db, embedding)
            results = db.search(embedding, top_k=5)
            assert len(results) == 1
            assert results[0]["chunk_text"] == "hello world"
            assert results[0]["session_id"] == "abc-123"


def test_get_all_files_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        with SearchDB(db_path) as db:
            files = db.get_all_files()
            assert files == {}


def test_delete_chunks_by_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        with SearchDB(db_path) as db:
            embedding = [0.1] * 384
            _insert_one(db, embedding)
            db.delete_chunks_by_file("/path.jsonl")
            db.commit()
            results = db.search(embedding, top_k=5)
            assert len(results) == 0


def test_stats():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        with SearchDB(db_path) as db:
            stats = db.get_stats()
            assert stats["total_chunks"] == 0
            assert stats["total_files"] == 0
