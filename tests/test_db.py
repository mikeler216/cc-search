import os
import sqlite3
import tempfile

from cc_search.db import SearchDB


def test_create_tables():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        db = SearchDB(db_path)
        conn = db.get_connection()
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in cursor.fetchall()]
        assert "chunks" in tables
        assert "files" in tables
        db.close()


def test_insert_and_get_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        db = SearchDB(db_path)
        db.upsert_file("/some/path.jsonl", last_modified=1000.0, last_indexed=1001.0)
        row = db.get_file("/some/path.jsonl")
        assert row is not None
        assert row["last_modified"] == 1000.0
        assert row["last_indexed"] == 1001.0
        db.close()


def test_insert_chunk_and_search_vec():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        db = SearchDB(db_path)
        embedding = [0.1] * 384
        db.insert_chunk(
            file_path="/path.jsonl",
            session_id="abc-123",
            project="my-project",
            role="user",
            chunk_text="hello world",
            turn_index=0,
            created_at=1000.0,
            embedding=embedding,
        )
        results = db.search(embedding, top_k=5)
        assert len(results) == 1
        assert results[0]["chunk_text"] == "hello world"
        assert results[0]["session_id"] == "abc-123"
        db.close()


def test_get_all_files_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        db = SearchDB(db_path)
        files = db.get_all_files()
        assert files == {}
        db.close()


def test_delete_chunks_by_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        db = SearchDB(db_path)
        embedding = [0.1] * 384
        db.insert_chunk(
            file_path="/path.jsonl",
            session_id="abc-123",
            project="proj",
            role="user",
            chunk_text="hello",
            turn_index=0,
            created_at=1000.0,
            embedding=embedding,
        )
        db.delete_chunks_by_file("/path.jsonl")
        results = db.search(embedding, top_k=5)
        assert len(results) == 0
        db.close()


def test_stats():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        db = SearchDB(db_path)
        stats = db.get_stats()
        assert stats["total_chunks"] == 0
        assert stats["total_files"] == 0
        db.close()
