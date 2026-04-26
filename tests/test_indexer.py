import json
import os
import tempfile

from cc_search.indexer import parse_jsonl_file, extract_text_from_message, Indexer
from cc_search.db import SearchDB


def _make_jsonl(tmpdir, filename, lines):
    path = os.path.join(tmpdir, filename)
    with open(path, "w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")
    return path


def test_extract_text_from_string_content():
    msg = {"role": "user", "content": "hello world"}
    assert extract_text_from_message(msg) == "hello world"


def test_extract_text_from_list_content():
    msg = {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "let me think"},
            {"type": "text", "text": "Here is the answer."},
            {"type": "tool_use", "name": "Bash", "id": "123"},
        ],
    }
    assert extract_text_from_message(msg) == "Here is the answer."


def test_extract_text_skips_tool_results():
    msg = {
        "role": "user",
        "content": [{"type": "tool_result", "content": "some output"}],
    }
    assert extract_text_from_message(msg) == ""


def test_parse_jsonl_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        lines = [
            {"type": "permission-mode", "permissionMode": "default", "sessionId": "sess-1"},
            {
                "type": "user",
                "message": {"role": "user", "content": "How do I fix this bug?"},
                "sessionId": "sess-1",
                "timestamp": 1000,
            },
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Try checking the logs."}],
                },
                "sessionId": "sess-1",
                "timestamp": 1001,
            },
        ]
        path = _make_jsonl(tmpdir, "sess-1.jsonl", lines)
        turns = parse_jsonl_file(path)
        assert len(turns) == 2
        assert turns[0]["role"] == "user"
        assert turns[0]["text"] == "How do I fix this bug?"
        assert turns[0]["session_id"] == "sess-1"
        assert turns[1]["role"] == "assistant"
        assert turns[1]["text"] == "Try checking the logs."


def test_indexer_indexes_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        projects_dir = os.path.join(tmpdir, "projects", "-Users-test-myproject")
        os.makedirs(projects_dir)
        lines = [
            {
                "type": "user",
                "message": {"role": "user", "content": "What is semantic search?"},
                "sessionId": "sess-1",
                "timestamp": 1000,
            },
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Semantic search uses embeddings to find similar meaning."}
                    ],
                },
                "sessionId": "sess-1",
                "timestamp": 1001,
            },
        ]
        _make_jsonl(projects_dir, "sess-1.jsonl", lines)
        db_path = os.path.join(tmpdir, "test.db")
        indexer = Indexer(db_path=db_path, claude_dir=tmpdir)
        indexer.index()
        stats = indexer.db.get_stats()
        assert stats["total_chunks"] == 2
        assert stats["total_files"] == 1
        indexer.db.close()


def test_indexer_incremental_skips_unchanged():
    with tempfile.TemporaryDirectory() as tmpdir:
        projects_dir = os.path.join(tmpdir, "projects", "-Users-test-proj")
        os.makedirs(projects_dir)
        lines = [
            {
                "type": "user",
                "message": {"role": "user", "content": "Hello"},
                "sessionId": "s1",
                "timestamp": 1000,
            },
        ]
        _make_jsonl(projects_dir, "s1.jsonl", lines)
        db_path = os.path.join(tmpdir, "test.db")
        indexer = Indexer(db_path=db_path, claude_dir=tmpdir)
        indexer.index()
        assert indexer.db.get_stats()["total_chunks"] == 1
        # Index again — should not add duplicates
        indexer.index()
        assert indexer.db.get_stats()["total_chunks"] == 1
        indexer.db.close()
