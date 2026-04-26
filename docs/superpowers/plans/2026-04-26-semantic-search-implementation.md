# Conversation Semantic Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `cc-search`, a local Python CLI + Claude Code skill that provides sub-second semantic search over Claude Code conversation history.

**Architecture:** Python CLI managed by `uv`, using `sentence-transformers` (`all-MiniLM-L6-v2`) for embeddings, `sqlite-vec` for vector storage/search in SQLite, and `click` for the CLI. Conversations are read from `~/.claude/projects/**/*.jsonl`, chunked per-turn, embedded, and stored. Search embeds the query and runs cosine similarity via `sqlite-vec`.

**Tech Stack:** Python 3.11+, uv, sentence-transformers, sqlite-vec, click

---

## File Structure

```
claude-conversation-semantic-search/
├── pyproject.toml                  # uv project config, deps, entry point
├── src/
│   └── cc_search/
│       ├── __init__.py             # package init, version
│       ├── db.py                   # SQLite + sqlite-vec schema, connection, queries
│       ├── chunker.py              # text chunking with overlap for long turns
│       ├── indexer.py              # JSONL parsing, embedding, DB insertion
│       ├── searcher.py             # query embedding + vector search
│       └── cli.py                  # click CLI: index, query, status
├── tests/
│   ├── __init__.py
│   ├── test_db.py
│   ├── test_chunker.py
│   ├── test_indexer.py
│   ├── test_searcher.py
│   └── test_cli.py
└── skill/
    └── search-history.md           # Claude Code skill
```

---

### Task 1: Project Scaffolding with uv

**Files:**
- Create: `pyproject.toml`
- Create: `src/cc_search/__init__.py`

- [ ] **Step 1: Initialize uv project**

```bash
cd /Users/michaeller/IdeaProjects/claude-converstaion-symentic-search
uv init --lib --name cc-search
```

This creates `pyproject.toml` and `src/cc_search/__init__.py`. If `uv init` complains about existing files, we'll create them manually.

- [ ] **Step 2: Set up pyproject.toml**

Replace the generated `pyproject.toml` with:

```toml
[project]
name = "cc-search"
version = "0.1.0"
description = "Semantic search over Claude Code conversation history"
requires-python = ">=3.11"
dependencies = [
    "sentence-transformers>=3.0.0",
    "sqlite-vec>=0.1.1",
    "click>=8.1.0",
]

[project.scripts]
cc-search = "cc_search.cli:cli"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
testpaths = ["tests"]

[dependency-groups]
dev = [
    "pytest>=8.0.0",
]
```

- [ ] **Step 3: Create __init__.py**

```python
__version__ = "0.1.0"
```

- [ ] **Step 4: Create test directory**

```bash
mkdir -p tests
touch tests/__init__.py
```

- [ ] **Step 5: Sync dependencies**

```bash
uv sync
```

- [ ] **Step 6: Verify uv setup**

```bash
uv run python -c "import cc_search; print(cc_search.__version__)"
```

Expected: `0.1.0`

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/ tests/ uv.lock
git commit -m "feat: scaffold uv project with dependencies"
```

---

### Task 2: Database Layer (db.py)

**Files:**
- Create: `src/cc_search/db.py`
- Create: `tests/test_db.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_db.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_db.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'cc_search.db'`

- [ ] **Step 3: Implement db.py**

Create `src/cc_search/db.py`:

```python
import os
import sqlite3
import struct
from typing import Any

import sqlite_vec


def _serialize_f32(vector: list[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


class SearchDB:
    def __init__(self, db_path: str):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)
        self._create_tables()

    def _create_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS files (
                path TEXT PRIMARY KEY,
                last_modified REAL,
                last_indexed REAL
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT,
                session_id TEXT,
                project TEXT,
                role TEXT,
                chunk_text TEXT,
                turn_index INTEGER,
                created_at REAL
            );
        """)
        self._conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0 (
                id INTEGER PRIMARY KEY,
                embedding FLOAT[384]
            )
        """)
        self._conn.commit()

    def get_connection(self) -> sqlite3.Connection:
        return self._conn

    def close(self):
        self._conn.close()

    def upsert_file(self, path: str, last_modified: float, last_indexed: float):
        self._conn.execute(
            "INSERT OR REPLACE INTO files (path, last_modified, last_indexed) VALUES (?, ?, ?)",
            (path, last_modified, last_indexed),
        )
        self._conn.commit()

    def get_file(self, path: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM files WHERE path = ?", (path,)
        ).fetchone()
        if row is None:
            return None
        return dict(row)

    def get_all_files(self) -> dict[str, dict[str, Any]]:
        rows = self._conn.execute("SELECT * FROM files").fetchall()
        return {row["path"]: dict(row) for row in rows}

    def insert_chunk(
        self,
        file_path: str,
        session_id: str,
        project: str,
        role: str,
        chunk_text: str,
        turn_index: int,
        created_at: float,
        embedding: list[float],
    ):
        cursor = self._conn.execute(
            """INSERT INTO chunks (file_path, session_id, project, role, chunk_text, turn_index, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (file_path, session_id, project, role, chunk_text, turn_index, created_at),
        )
        chunk_id = cursor.lastrowid
        self._conn.execute(
            "INSERT INTO chunks_vec (id, embedding) VALUES (?, ?)",
            (chunk_id, _serialize_f32(embedding)),
        )
        self._conn.commit()

    def insert_chunks_batch(
        self, chunks: list[dict[str, Any]], embeddings: list[list[float]]
    ):
        for chunk, embedding in zip(chunks, embeddings):
            cursor = self._conn.execute(
                """INSERT INTO chunks (file_path, session_id, project, role, chunk_text, turn_index, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    chunk["file_path"],
                    chunk["session_id"],
                    chunk["project"],
                    chunk["role"],
                    chunk["chunk_text"],
                    chunk["turn_index"],
                    chunk["created_at"],
                ),
            )
            chunk_id = cursor.lastrowid
            self._conn.execute(
                "INSERT INTO chunks_vec (id, embedding) VALUES (?, ?)",
                (chunk_id, _serialize_f32(embedding)),
            )
        self._conn.commit()

    def delete_chunks_by_file(self, file_path: str):
        chunk_ids = self._conn.execute(
            "SELECT id FROM chunks WHERE file_path = ?", (file_path,)
        ).fetchall()
        for row in chunk_ids:
            self._conn.execute("DELETE FROM chunks_vec WHERE id = ?", (row["id"],))
        self._conn.execute("DELETE FROM chunks WHERE file_path = ?", (file_path,))
        self._conn.commit()

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        project: str | None = None,
        role: str | None = None,
    ) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT v.id, v.distance, c.*
            FROM chunks_vec v
            INNER JOIN chunks c ON c.id = v.id
            WHERE v.embedding MATCH ?
            AND k = ?
            ORDER BY v.distance
            """,
            (_serialize_f32(query_embedding), top_k * 5 if (project or role) else top_k),
        ).fetchall()

        results = []
        for row in rows:
            r = dict(row)
            if project and r["project"] != project:
                continue
            if role and r["role"] != role:
                continue
            r["score"] = 1.0 - r["distance"]
            results.append(r)
            if len(results) >= top_k:
                break
        return results

    def get_stats(self) -> dict[str, Any]:
        total_chunks = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        total_files = self._conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        return {"total_chunks": total_chunks, "total_files": total_files}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_db.py -v
```

Expected: All 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cc_search/db.py tests/test_db.py
git commit -m "feat: add database layer with sqlite-vec vector search"
```

---

### Task 3: Chunker (chunker.py)

**Files:**
- Create: `src/cc_search/chunker.py`
- Create: `tests/test_chunker.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_chunker.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_chunker.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'cc_search.chunker'`

- [ ] **Step 3: Implement chunker.py**

Create `src/cc_search/chunker.py`:

```python
def chunk_text(
    text: str, max_tokens: int = 400, overlap_tokens: int = 100
) -> list[str]:
    text = text.strip()
    if not text:
        return []

    words = text.split()
    if len(words) <= max_tokens:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_tokens, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break
        start += max_tokens - overlap_tokens

    return chunks
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_chunker.py -v
```

Expected: All 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cc_search/chunker.py tests/test_chunker.py
git commit -m "feat: add text chunker with overlap windowing"
```

---

### Task 4: Indexer (indexer.py)

**Files:**
- Create: `src/cc_search/indexer.py`
- Create: `tests/test_indexer.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_indexer.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_indexer.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'cc_search.indexer'`

- [ ] **Step 3: Implement indexer.py**

Create `src/cc_search/indexer.py`:

```python
import glob
import json
import os
import time
from typing import Any

from sentence_transformers import SentenceTransformer

from cc_search.chunker import chunk_text
from cc_search.db import SearchDB

MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_CLAUDE_DIR = os.path.expanduser("~/.claude")
DEFAULT_DB_PATH = os.path.expanduser("~/.claude/search-index/conversations.db")


def extract_text_from_message(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        texts = []
        for block in content:
            if block.get("type") == "text":
                texts.append(block.get("text", ""))
        return " ".join(texts).strip()
    return ""


def _project_name_from_path(file_path: str) -> str:
    parts = file_path.split("/projects/")
    if len(parts) < 2:
        return "unknown"
    project_dir = parts[1].split("/")[0]
    return project_dir.replace("-", "/").lstrip("/")


def parse_jsonl_file(file_path: str) -> list[dict[str, Any]]:
    turns = []
    turn_index = 0
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            entry_type = entry.get("type", "")
            if entry_type not in ("user", "assistant"):
                continue
            message = entry.get("message", {})
            text = extract_text_from_message(message)
            if not text:
                continue
            turns.append(
                {
                    "role": entry_type,
                    "text": text,
                    "session_id": entry.get("sessionId", ""),
                    "timestamp": entry.get("timestamp", 0),
                    "turn_index": turn_index,
                }
            )
            turn_index += 1
    return turns


class Indexer:
    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        claude_dir: str = DEFAULT_CLAUDE_DIR,
        model_name: str = MODEL_NAME,
    ):
        self.db = SearchDB(db_path)
        self.claude_dir = claude_dir
        self._model = None
        self._model_name = model_name

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def _find_jsonl_files(self) -> list[str]:
        projects_dir = os.path.join(self.claude_dir, "projects")
        if not os.path.isdir(projects_dir):
            return []
        return glob.glob(os.path.join(projects_dir, "**", "*.jsonl"), recursive=True)

    def index(self, full: bool = False):
        files = self._find_jsonl_files()
        indexed_files = self.db.get_all_files()

        for file_path in files:
            mtime = os.path.getmtime(file_path)

            if not full and file_path in indexed_files:
                if indexed_files[file_path]["last_modified"] >= mtime:
                    continue

            self._index_file(file_path, mtime)

    def _index_file(self, file_path: str, mtime: float):
        self.db.delete_chunks_by_file(file_path)

        turns = parse_jsonl_file(file_path)
        if not turns:
            self.db.upsert_file(file_path, last_modified=mtime, last_indexed=time.time())
            return

        project = _project_name_from_path(file_path)

        all_chunks = []
        all_texts = []
        for turn in turns:
            text_chunks = chunk_text(turn["text"])
            for chunk in text_chunks:
                all_chunks.append(
                    {
                        "file_path": file_path,
                        "session_id": turn["session_id"],
                        "project": project,
                        "role": turn["role"],
                        "chunk_text": chunk,
                        "turn_index": turn["turn_index"],
                        "created_at": turn["timestamp"],
                    }
                )
                all_texts.append(chunk)

        if all_texts:
            embeddings = self.model.encode(all_texts).tolist()
            self.db.insert_chunks_batch(all_chunks, embeddings)

        self.db.upsert_file(file_path, last_modified=mtime, last_indexed=time.time())
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_indexer.py -v
```

Expected: All 6 tests PASS (first run downloads the model, may take 30-60 seconds).

- [ ] **Step 5: Commit**

```bash
git add src/cc_search/indexer.py tests/test_indexer.py
git commit -m "feat: add indexer for JSONL parsing and embedding"
```

---

### Task 5: Searcher (searcher.py)

**Files:**
- Create: `src/cc_search/searcher.py`
- Create: `tests/test_searcher.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_searcher.py`:

```python
import os
import tempfile

from cc_search.searcher import Searcher
from cc_search.db import SearchDB


def _seed_db(db: SearchDB, model):
    texts = [
        ("How do I set up JWT auth middleware?", "user"),
        ("Here is how to configure JWT validation.", "assistant"),
        ("Fix the database migration bug", "user"),
        ("The migration failed because of a missing column.", "assistant"),
        ("How does React state management work?", "user"),
    ]
    for i, (text, role) in enumerate(texts):
        embedding = model.encode(text).tolist()
        db.insert_chunk(
            file_path="/test.jsonl",
            session_id="sess-1",
            project="Users/test/myproject",
            role=role,
            chunk_text=text,
            turn_index=i,
            created_at=1000.0 + i,
            embedding=embedding,
        )


def test_search_returns_relevant_results():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        searcher = Searcher(db_path=db_path)
        _seed_db(searcher.db, searcher.model)
        results = searcher.search("authentication setup")
        assert len(results) > 0
        top_text = results[0]["chunk_text"]
        assert "JWT" in top_text or "auth" in top_text


def test_search_top_k():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        searcher = Searcher(db_path=db_path)
        _seed_db(searcher.db, searcher.model)
        results = searcher.search("anything", top_k=2)
        assert len(results) <= 2


def test_search_empty_db():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        searcher = Searcher(db_path=db_path)
        results = searcher.search("hello")
        assert results == []


def test_search_with_resume_command():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        searcher = Searcher(db_path=db_path)
        _seed_db(searcher.db, searcher.model)
        results = searcher.search("database migration")
        assert len(results) > 0
        assert results[0]["resume_command"] == "claude --resume sess-1"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_searcher.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'cc_search.searcher'`

- [ ] **Step 3: Implement searcher.py**

Create `src/cc_search/searcher.py`:

```python
from typing import Any

from sentence_transformers import SentenceTransformer

from cc_search.db import SearchDB
from cc_search.indexer import DEFAULT_DB_PATH, MODEL_NAME


class Searcher:
    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        model_name: str = MODEL_NAME,
    ):
        self.db = SearchDB(db_path)
        self._model = None
        self._model_name = model_name

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def search(
        self,
        query: str,
        top_k: int = 5,
        project: str | None = None,
        role: str | None = None,
    ) -> list[dict[str, Any]]:
        query_embedding = self.model.encode(query).tolist()
        results = self.db.search(
            query_embedding, top_k=top_k, project=project, role=role
        )
        for r in results:
            r["resume_command"] = f"claude --resume {r['session_id']}"
        return results
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_searcher.py -v
```

Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cc_search/searcher.py tests/test_searcher.py
git commit -m "feat: add searcher with query embedding and resume links"
```

---

### Task 6: CLI (cli.py)

**Files:**
- Create: `src/cc_search/cli.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_cli.py`:

```python
import json
import os
import tempfile

from click.testing import CliRunner

from cc_search.cli import cli


def _setup_test_env(tmpdir):
    claude_dir = os.path.join(tmpdir, "claude")
    projects_dir = os.path.join(claude_dir, "projects", "-Users-test-proj")
    os.makedirs(projects_dir)
    db_path = os.path.join(tmpdir, "test.db")

    lines = [
        {
            "type": "user",
            "message": {"role": "user", "content": "How do I configure JWT auth?"},
            "sessionId": "sess-abc",
            "timestamp": 1000,
        },
        {
            "type": "assistant",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Set up JWT middleware like this."}],
            },
            "sessionId": "sess-abc",
            "timestamp": 1001,
        },
    ]
    jsonl_path = os.path.join(projects_dir, "sess-abc.jsonl")
    with open(jsonl_path, "w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")

    return claude_dir, db_path


def test_index_command():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        claude_dir, db_path = _setup_test_env(tmpdir)
        result = runner.invoke(
            cli, ["index", "--db-path", db_path, "--claude-dir", claude_dir]
        )
        assert result.exit_code == 0
        assert "Indexed" in result.output


def test_query_command():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        claude_dir, db_path = _setup_test_env(tmpdir)
        runner.invoke(
            cli, ["index", "--db-path", db_path, "--claude-dir", claude_dir]
        )
        result = runner.invoke(
            cli, ["query", "JWT auth", "--db-path", db_path, "--top", "3"]
        )
        assert result.exit_code == 0
        assert "claude --resume" in result.output


def test_status_command():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        claude_dir, db_path = _setup_test_env(tmpdir)
        runner.invoke(
            cli, ["index", "--db-path", db_path, "--claude-dir", claude_dir]
        )
        result = runner.invoke(cli, ["status", "--db-path", db_path])
        assert result.exit_code == 0
        assert "chunks" in result.output.lower() or "Chunks" in result.output


def test_query_empty_db():
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "empty.db")
        result = runner.invoke(cli, ["query", "hello", "--db-path", db_path])
        assert result.exit_code == 0
        assert "No results" in result.output or "no results" in result.output.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/test_cli.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'cc_search.cli'`

- [ ] **Step 3: Implement cli.py**

Create `src/cc_search/cli.py`:

```python
import time

import click

from cc_search.indexer import DEFAULT_DB_PATH, DEFAULT_CLAUDE_DIR, Indexer
from cc_search.searcher import Searcher


@click.group()
def cli():
    pass


@cli.command()
@click.option("--db-path", default=DEFAULT_DB_PATH, help="Path to the search database")
@click.option("--claude-dir", default=DEFAULT_CLAUDE_DIR, help="Path to ~/.claude directory")
@click.option("--full", is_flag=True, help="Force full reindex")
@click.option("--watch", is_flag=True, help="Watch for changes and reindex")
def index(db_path, claude_dir, full, watch):
    indexer = Indexer(db_path=db_path, claude_dir=claude_dir)
    if watch:
        click.echo("Watching for changes... (Ctrl+C to stop)")
        while True:
            indexer.index(full=False)
            time.sleep(30)
    else:
        indexer.index(full=full)
        stats = indexer.db.get_stats()
        click.echo(
            f"Indexed {stats['total_chunks']} chunks from {stats['total_files']} files."
        )
    indexer.db.close()


@cli.command()
@click.argument("query_text")
@click.option("--db-path", default=DEFAULT_DB_PATH, help="Path to the search database")
@click.option("--top", default=5, help="Number of results to return")
@click.option("--project", default=None, help="Filter by project name")
@click.option("--role", default=None, type=click.Choice(["user", "assistant"]), help="Filter by role")
def query(query_text, db_path, top, project, role):
    searcher = Searcher(db_path=db_path)
    results = searcher.search(query_text, top_k=top, project=project, role=role)

    if not results:
        click.echo("No results found.")
        return

    for i, r in enumerate(results, 1):
        score = r.get("score", 0)
        click.echo(f"── Result {i} (score: {score:.2f}) {'─' * 30}")
        click.echo(f"Project: {r['project']}")
        click.echo(f"Role:    {r['role']}")
        click.echo(f"Turn:    {r['turn_index']}")
        click.echo()
        text = r["chunk_text"]
        if len(text) > 300:
            text = text[:300] + "..."
        for line in text.split("\n"):
            click.echo(f"  {line}")
        click.echo()
        click.echo(f"  → {r['resume_command']}")
        click.echo()

    searcher.db.close()


@cli.command()
@click.option("--db-path", default=DEFAULT_DB_PATH, help="Path to the search database")
def status(db_path):
    from cc_search.db import SearchDB
    import os

    db = SearchDB(db_path)
    stats = db.get_stats()
    db_size = os.path.getsize(db_path) if os.path.exists(db_path) else 0
    click.echo(f"Chunks:  {stats['total_chunks']}")
    click.echo(f"Files:   {stats['total_files']}")
    click.echo(f"DB size: {db_size / 1024:.1f} KB")
    db.close()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/test_cli.py -v
```

Expected: All 4 tests PASS.

- [ ] **Step 5: Verify CLI entry point works**

```bash
uv run cc-search --help
uv run cc-search index --help
uv run cc-search query --help
uv run cc-search status --help
```

Expected: Each shows help text with the documented options.

- [ ] **Step 6: Commit**

```bash
git add src/cc_search/cli.py tests/test_cli.py
git commit -m "feat: add click CLI with index, query, and status commands"
```

---

### Task 7: Claude Code Skill

**Files:**
- Create: `skill/search-history.md`

- [ ] **Step 1: Create skill file**

Create `skill/search-history.md`:

```markdown
---
name: search-history
description: Semantic search over past Claude Code conversations. Use when the user wants to find a previous conversation, recall how something was done before, or search their history. Triggers on "do you remember", "find that conversation", "search history", "when did we", or /search-history.
---

Run `cc-search query "<user's search terms>" --top 5` via Bash.

If the command is not found, tell the user:
> `cc-search` is not installed. Install it with:
> ```
> uv tool install /Users/michaeller/IdeaProjects/claude-converstaion-symentic-search
> ```

If the output says "No results found", suggest the user run `cc-search index` to build or update the index.

Otherwise, present each result as:
- The matching text snippet
- Project name and whether it was a user message or assistant response
- The resume command so they can jump back into that conversation

Keep presentation concise — just the essential info from each result.
```

- [ ] **Step 2: Commit**

```bash
mkdir -p skill
git add skill/search-history.md
git commit -m "feat: add Claude Code search-history skill"
```

---

### Task 8: Install and End-to-End Test

- [ ] **Step 1: Install the tool**

```bash
cd /Users/michaeller/IdeaProjects/claude-converstaion-symentic-search
uv tool install . --force
```

- [ ] **Step 2: Verify cc-search is available**

```bash
cc-search --help
```

Expected: Shows help with `index`, `query`, `status` subcommands.

- [ ] **Step 3: Run the indexer on real data**

```bash
cc-search index
```

Expected: Output like `Indexed N chunks from M files.` (first run downloads model, may take 30-60s).

- [ ] **Step 4: Check status**

```bash
cc-search status
```

Expected: Shows non-zero chunk and file counts.

- [ ] **Step 5: Run a test query**

```bash
cc-search query "semantic search"
```

Expected: Returns results with scores, project names, and `claude --resume` commands.

- [ ] **Step 6: Run full test suite**

```bash
uv run pytest tests/ -v
```

Expected: All tests pass.

- [ ] **Step 7: Install the skill**

```bash
ln -sf /Users/michaeller/IdeaProjects/claude-converstaion-symentic-search/skill/search-history.md ~/.claude/skills/search-history.md
```

- [ ] **Step 8: Commit any final adjustments**

```bash
git add -A
git commit -m "chore: finalize installation and e2e verification"
```
