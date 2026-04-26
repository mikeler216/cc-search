import os
import sqlite3
import struct

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
                embedding FLOAT[384] distance_metric=cosine
            )
        """)
        self._conn.commit()

    def close(self):
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def upsert_file(self, path: str, last_modified: float, last_indexed: float):
        self._conn.execute(
            "INSERT OR REPLACE INTO files (path, last_modified, last_indexed) VALUES (?, ?, ?)",
            (path, last_modified, last_indexed),
        )

    def get_file(self, path: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM files WHERE path = ?", (path,)
        ).fetchone()
        if row is None:
            return None
        return dict(row)

    def get_all_files(self) -> dict[str, dict]:
        rows = self._conn.execute("SELECT path, last_modified FROM files").fetchall()
        return {row["path"]: dict(row) for row in rows}

    def insert_chunks_batch(self, chunks: list[dict], embeddings: list[list[float]]):
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
            self._conn.execute(
                "INSERT INTO chunks_vec (id, embedding) VALUES (?, ?)",
                (cursor.lastrowid, _serialize_f32(embedding)),
            )

    def delete_chunks_by_file(self, file_path: str):
        self._conn.execute(
            "DELETE FROM chunks_vec WHERE id IN (SELECT id FROM chunks WHERE file_path = ?)",
            (file_path,),
        )
        self._conn.execute("DELETE FROM chunks WHERE file_path = ?", (file_path,))

    def commit(self):
        self._conn.commit()

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        project: str | None = None,
        role: str | None = None,
    ) -> list[dict]:
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
            if project and not r["project"].startswith(project):
                continue
            if role and r["role"] != role:
                continue
            r["score"] = 1.0 - r["distance"]
            results.append(r)
            if len(results) >= top_k:
                break
        return results

    def get_stats(self) -> dict:
        total_chunks = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        total_files = self._conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
        return {"total_chunks": total_chunks, "total_files": total_files}
