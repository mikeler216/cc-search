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
