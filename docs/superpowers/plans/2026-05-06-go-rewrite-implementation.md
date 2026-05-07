# cc-search Go Rewrite — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite cc-search from Python to Go as a single static binary with embedded ONNX model for zero-dependency distribution.

**Architecture:** Four internal packages (`store`, `embedding`, `ingest`, `search`) with a clean DAG dependency graph. CGo bridges to ONNX Runtime for inference and SQLite + sqlite-vec for vector search. The ONNX model and WordPiece vocab are embedded in the binary via `//go:embed`.

**Tech Stack:** Go 1.22+, `github.com/yalue/onnxruntime_go`, `github.com/mattn/go-sqlite3`, `github.com/asg017/sqlite-vec-go-bindings/cgo`, `github.com/spf13/cobra`, `github.com/fsnotify/fsnotify`

---

## Prerequisites

Before starting, the developer needs:

1. **Go 1.22+** — `brew install go` or https://go.dev/dl
2. **ONNX Runtime** — `brew install onnxruntime` (macOS). On Linux, download from https://github.com/microsoft/onnxruntime/releases and place `libonnxruntime.so` in `/usr/local/lib`.
3. **Python 3.11+** (one-time, for model export) — already present in this project's env

The ONNX Runtime shared library path varies by platform:
- macOS arm64: `/opt/homebrew/lib/libonnxruntime.dylib`
- macOS amd64: `/usr/local/lib/libonnxruntime.dylib`
- Linux: `/usr/local/lib/libonnxruntime.so`

---

### Task 1: Project Scaffolding & Asset Export

**Files:**
- Create: `go.mod`
- Create: `cmd/cc-search/main.go`
- Create: `internal/assets/assets.go`
- Create: `internal/assets/.gitkeep`
- Create: `scripts/export-model.py`
- Create: `Makefile`
- Modify: `.gitignore`

This task sets up the Go module, directory structure, and exports the ONNX model + vocab from the existing Python environment.

- [ ] **Step 1: Initialize Go module and create directory structure**

```bash
cd /Users/michaeller/IdeaProjects/claude-converstaion-symentic-search
go mod init github.com/mikeler216/cc-search
mkdir -p cmd/cc-search internal/assets internal/store internal/embedding internal/ingest internal/search
```

- [ ] **Step 2: Create the model export script**

Create `scripts/export-model.py`:

```python
#!/usr/bin/env python3
"""Export all-MiniLM-L6-v2 to ONNX format and extract vocab.txt."""
import os
import shutil
from pathlib import Path

def main():
    assets_dir = Path(__file__).parent.parent / "internal" / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    model_path = assets_dir / "model.onnx"
    vocab_path = assets_dir / "vocab.txt"

    if model_path.exists() and vocab_path.exists():
        print("Assets already exist, skipping export.")
        return

    print("Exporting all-MiniLM-L6-v2 to ONNX...")

    from optimum.exporters.onnx import main_export
    from transformers import AutoTokenizer

    export_dir = "/tmp/cc-search-onnx-export"
    main_export(
        "sentence-transformers/all-MiniLM-L6-v2",
        output=export_dir,
        task="feature-extraction",
    )

    shutil.copy(os.path.join(export_dir, "model.onnx"), model_path)
    print(f"Wrote {model_path} ({model_path.stat().st_size / 1024 / 1024:.1f} MB)")

    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    vocab = tokenizer.get_vocab()
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    with open(vocab_path, "w") as f:
        for token, _ in sorted_vocab:
            f.write(token + "\n")
    print(f"Wrote {vocab_path} ({len(sorted_vocab)} tokens)")

if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the export script to produce assets**

```bash
uv pip install optimum[exporters] transformers
uv run python scripts/export-model.py
```

Expected: `internal/assets/model.onnx` (~30MB) and `internal/assets/vocab.txt` (~230KB) are created.

Verify:

```bash
ls -lh internal/assets/model.onnx internal/assets/vocab.txt
```

- [ ] **Step 4: Create the Go embed file for assets**

Create `internal/assets/assets.go`:

```go
package assets

import _ "embed"

//go:embed model.onnx
var ModelONNX []byte

//go:embed vocab.txt
var VocabTxt []byte
```

- [ ] **Step 5: Create the minimal main.go**

Create `cmd/cc-search/main.go`:

```go
package main

import (
	"fmt"
	"os"
)

func main() {
	fmt.Fprintln(os.Stderr, "cc-search: not yet implemented")
	os.Exit(1)
}
```

- [ ] **Step 6: Create Makefile**

Create `Makefile`:

```makefile
.PHONY: build test assets clean

BINARY := cc-search
ONNX_LIB ?= $(shell \
	if [ -f /opt/homebrew/lib/libonnxruntime.dylib ]; then echo /opt/homebrew/lib/libonnxruntime.dylib; \
	elif [ -f /usr/local/lib/libonnxruntime.dylib ]; then echo /usr/local/lib/libonnxruntime.dylib; \
	elif [ -f /usr/local/lib/libonnxruntime.so ]; then echo /usr/local/lib/libonnxruntime.so; \
	else echo ""; fi)

build: internal/assets/model.onnx
	CGO_ENABLED=1 go build -o $(BINARY) ./cmd/cc-search

test: internal/assets/model.onnx
	CGO_ENABLED=1 ONNX_LIB=$(ONNX_LIB) go test ./... -v -count=1

assets: internal/assets/model.onnx

internal/assets/model.onnx:
	python scripts/export-model.py

clean:
	rm -f $(BINARY)
```

- [ ] **Step 7: Update .gitignore**

Append to `.gitignore`:

```
# Go
cc-search
internal/assets/model.onnx
/vendor/
```

Note: `model.onnx` is gitignored because it's ~30MB. CI will run the export script. `vocab.txt` is checked in (small file).

- [ ] **Step 8: Install Go dependencies and verify build**

```bash
go get github.com/yalue/onnxruntime_go
go get github.com/mattn/go-sqlite3
go get github.com/spf13/cobra
go get github.com/fsnotify/fsnotify
go mod tidy
go build ./cmd/cc-search
```

Expected: compiles successfully, prints "cc-search: not yet implemented" when run.

- [ ] **Step 9: Commit**

```bash
git add go.mod go.sum cmd/ internal/assets/assets.go internal/assets/vocab.txt scripts/export-model.py Makefile .gitignore
git commit -m "feat: scaffold Go project with module, assets embed, and model export"
```

---

### Task 2: Store Package

**Files:**
- Create: `internal/store/store.go`
- Create: `internal/store/store_test.go`

This task implements the full SQLite + sqlite-vec persistence layer: schema creation, meta tracking, file tracking, chunk storage with embeddings, vector search with filtering, and stats.

- [ ] **Step 1: Write failing tests for schema creation and meta operations**

Create `internal/store/store_test.go`:

```go
package store

import (
	"os"
	"path/filepath"
	"testing"
)

func tempDB(t *testing.T) *DB {
	t.Helper()
	dir := t.TempDir()
	db, err := Open(filepath.Join(dir, "test.db"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	t.Cleanup(func() { db.Close() })
	return db
}

func TestOpenCreatesTablesAndMeta(t *testing.T) {
	db := tempDB(t)

	var count int
	err := db.conn.QueryRow("SELECT COUNT(*) FROM meta").Scan(&count)
	if err != nil {
		t.Fatalf("query meta: %v", err)
	}
	if count < 2 {
		t.Errorf("expected at least 2 meta rows (schema_version, model_hash), got %d", count)
	}
}

func TestCheckVersionMatch(t *testing.T) {
	db := tempDB(t)

	match, err := db.CheckVersion("test-hash")
	if err != nil {
		t.Fatalf("CheckVersion: %v", err)
	}
	if !match {
		t.Error("expected version match with the hash used at creation")
	}
}

func TestCheckVersionMismatch(t *testing.T) {
	db := tempDB(t)

	match, err := db.CheckVersion("different-hash")
	if err != nil {
		t.Fatalf("CheckVersion: %v", err)
	}
	if match {
		t.Error("expected version mismatch")
	}
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/michaeller/IdeaProjects/claude-converstaion-symentic-search
CGO_ENABLED=1 go test ./internal/store/ -v -run 'TestOpen|TestCheckVersion' -count=1
```

Expected: FAIL — `store` package does not exist yet.

- [ ] **Step 3: Implement schema creation and meta operations**

Create `internal/store/store.go`:

```go
package store

import (
	"database/sql"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"path/filepath"

	_ "github.com/asg017/sqlite-vec-go-bindings/cgo"
	_ "github.com/mattn/go-sqlite3"
)

const SchemaVersion = "1"

type DB struct {
	conn      *sql.DB
	modelHash string
}

type FileRecord struct {
	Path         string
	LastModified float64
	LastIndexed  float64
}

type Chunk struct {
	FilePath  string
	SessionID string
	Project   string
	Role      string
	Text      string
	TurnIndex int
	CreatedAt float64
}

type Result struct {
	Chunk
	Score         float32
	ResumeCommand string
}

func Open(dbPath string) (*DB, error) {
	if err := os.MkdirAll(filepath.Dir(dbPath), 0755); err != nil {
		return nil, err
	}
	conn, err := sql.Open("sqlite3", dbPath)
	if err != nil {
		return nil, err
	}
	db := &DB{conn: conn}
	if err := db.createTables(); err != nil {
		conn.Close()
		return nil, err
	}
	return db, nil
}

func (db *DB) Close() error {
	return db.conn.Close()
}

func (db *DB) createTables() error {
	stmts := []string{
		`CREATE TABLE IF NOT EXISTS meta (
			key   TEXT PRIMARY KEY,
			value TEXT
		)`,
		`CREATE TABLE IF NOT EXISTS files (
			path          TEXT PRIMARY KEY,
			last_modified REAL NOT NULL,
			last_indexed  REAL NOT NULL
		)`,
		`CREATE TABLE IF NOT EXISTS chunks (
			id         INTEGER PRIMARY KEY AUTOINCREMENT,
			file_path  TEXT NOT NULL,
			session_id TEXT NOT NULL,
			project    TEXT NOT NULL,
			role       TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
			chunk_text TEXT NOT NULL,
			turn_index INTEGER NOT NULL,
			created_at REAL
		)`,
		`CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_path)`,
		`CREATE INDEX IF NOT EXISTS idx_chunks_project ON chunks(project)`,
		`CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0 (
			id        INTEGER PRIMARY KEY,
			embedding FLOAT[384] distance_metric=cosine
		)`,
	}
	for _, s := range stmts {
		if _, err := db.conn.Exec(s); err != nil {
			return fmt.Errorf("exec %q: %w", s[:40], err)
		}
	}
	return nil
}

func (db *DB) SetMeta(key, value string) error {
	_, err := db.conn.Exec(
		"INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", key, value)
	return err
}

func (db *DB) GetMeta(key string) (string, error) {
	var value string
	err := db.conn.QueryRow("SELECT value FROM meta WHERE key = ?", key).Scan(&value)
	if err == sql.ErrNoRows {
		return "", nil
	}
	return value, err
}

func (db *DB) InitMeta(modelHash string) error {
	db.modelHash = modelHash
	if err := db.SetMeta("schema_version", SchemaVersion); err != nil {
		return err
	}
	return db.SetMeta("model_hash", modelHash)
}

func (db *DB) CheckVersion(modelHash string) (bool, error) {
	sv, err := db.GetMeta("schema_version")
	if err != nil {
		return false, err
	}
	mh, err := db.GetMeta("model_hash")
	if err != nil {
		return false, err
	}
	return sv == SchemaVersion && mh == modelHash, nil
}

func (db *DB) DropAll() error {
	stmts := []string{
		"DROP TABLE IF EXISTS chunks_vec",
		"DROP TABLE IF EXISTS chunks",
		"DROP TABLE IF EXISTS files",
		"DROP TABLE IF EXISTS meta",
	}
	for _, s := range stmts {
		if _, err := db.conn.Exec(s); err != nil {
			return err
		}
	}
	return db.createTables()
}

func serializeF32(vec []float32) []byte {
	buf := make([]byte, len(vec)*4)
	for i, v := range vec {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
	}
	return buf
}
```

Note: `Open` needs to be updated — after `createTables`, we need to initialize meta if it doesn't exist. We'll handle that in the CLI layer where we know the model hash. For now `Open` just creates tables.

- [ ] **Step 4: Run tests to verify they pass**

First we need to update `Open` so that tests work. Update the `tempDB` helper and `TestOpenCreatesTablesAndMeta`:

Update `internal/store/store_test.go` — replace `tempDB`:

```go
func tempDB(t *testing.T) *DB {
	t.Helper()
	dir := t.TempDir()
	db, err := Open(filepath.Join(dir, "test.db"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if err := db.InitMeta("test-hash"); err != nil {
		t.Fatalf("InitMeta: %v", err)
	}
	t.Cleanup(func() { db.Close() })
	return db
}
```

```bash
CGO_ENABLED=1 go test ./internal/store/ -v -run 'TestOpen|TestCheckVersion' -count=1
```

Expected: PASS

- [ ] **Step 5: Write failing tests for file tracking**

Append to `internal/store/store_test.go`:

```go
func TestUpsertAndGetFile(t *testing.T) {
	db := tempDB(t)

	if err := db.UpsertFile("/some/path.jsonl", 1000.0, 1001.0); err != nil {
		t.Fatalf("UpsertFile: %v", err)
	}
	rec, err := db.GetFile("/some/path.jsonl")
	if err != nil {
		t.Fatalf("GetFile: %v", err)
	}
	if rec == nil {
		t.Fatal("expected non-nil file record")
	}
	if rec.LastModified != 1000.0 {
		t.Errorf("LastModified = %f, want 1000.0", rec.LastModified)
	}
}

func TestGetFileNotFound(t *testing.T) {
	db := tempDB(t)

	rec, err := db.GetFile("/nonexistent")
	if err != nil {
		t.Fatalf("GetFile: %v", err)
	}
	if rec != nil {
		t.Error("expected nil for nonexistent file")
	}
}

func TestAllFiles(t *testing.T) {
	db := tempDB(t)

	db.UpsertFile("/a.jsonl", 1.0, 2.0)
	db.UpsertFile("/b.jsonl", 3.0, 4.0)

	files, err := db.AllFiles()
	if err != nil {
		t.Fatalf("AllFiles: %v", err)
	}
	if len(files) != 2 {
		t.Errorf("got %d files, want 2", len(files))
	}
}
```

- [ ] **Step 6: Implement file tracking methods**

Append to `internal/store/store.go`:

```go
func (db *DB) UpsertFile(path string, lastModified, lastIndexed float64) error {
	_, err := db.conn.Exec(
		"INSERT OR REPLACE INTO files (path, last_modified, last_indexed) VALUES (?, ?, ?)",
		path, lastModified, lastIndexed)
	return err
}

func (db *DB) GetFile(path string) (*FileRecord, error) {
	var rec FileRecord
	err := db.conn.QueryRow(
		"SELECT path, last_modified, last_indexed FROM files WHERE path = ?", path,
	).Scan(&rec.Path, &rec.LastModified, &rec.LastIndexed)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	return &rec, nil
}

func (db *DB) AllFiles() (map[string]FileRecord, error) {
	rows, err := db.conn.Query("SELECT path, last_modified, last_indexed FROM files")
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	files := make(map[string]FileRecord)
	for rows.Next() {
		var rec FileRecord
		if err := rows.Scan(&rec.Path, &rec.LastModified, &rec.LastIndexed); err != nil {
			return nil, err
		}
		files[rec.Path] = rec
	}
	return files, rows.Err()
}
```

- [ ] **Step 7: Run file tracking tests**

```bash
CGO_ENABLED=1 go test ./internal/store/ -v -run 'TestUpsert|TestGetFile|TestAllFiles' -count=1
```

Expected: PASS

- [ ] **Step 8: Write failing tests for chunk storage and vector search**

Append to `internal/store/store_test.go`:

```go
func makeChunk(filePath, sessionID, project, role, text string, turnIndex int) Chunk {
	return Chunk{
		FilePath:  filePath,
		SessionID: sessionID,
		Project:   project,
		Role:      role,
		Text:      text,
		TurnIndex: turnIndex,
		CreatedAt: 1000.0,
	}
}

func TestInsertChunksAndSearch(t *testing.T) {
	db := tempDB(t)

	embedding := make([]float32, 384)
	for i := range embedding {
		embedding[i] = 0.1
	}
	chunks := []Chunk{makeChunk("/test.jsonl", "sess-1", "myproject", "user", "hello world", 0)}
	embeddings := [][]float32{embedding}

	if err := db.InsertChunks(chunks, embeddings); err != nil {
		t.Fatalf("InsertChunks: %v", err)
	}

	results, err := db.Search(embedding, 5, "", "")
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("got %d results, want 1", len(results))
	}
	if results[0].Text != "hello world" {
		t.Errorf("Text = %q, want %q", results[0].Text, "hello world")
	}
	if results[0].SessionID != "sess-1" {
		t.Errorf("SessionID = %q, want %q", results[0].SessionID, "sess-1")
	}
}

func TestSearchFilterByProject(t *testing.T) {
	db := tempDB(t)

	emb := make([]float32, 384)
	for i := range emb {
		emb[i] = 0.1
	}
	chunks := []Chunk{
		makeChunk("/a.jsonl", "s1", "-Users-test-projA", "user", "project A text", 0),
		makeChunk("/b.jsonl", "s2", "-Users-test-projB", "user", "project B text", 0),
	}
	db.InsertChunks(chunks, [][]float32{emb, emb})

	results, err := db.Search(emb, 5, "-Users-test-projA", "")
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(results) != 1 {
		t.Errorf("got %d results, want 1", len(results))
	}
}

func TestSearchFilterByRole(t *testing.T) {
	db := tempDB(t)

	emb := make([]float32, 384)
	for i := range emb {
		emb[i] = 0.1
	}
	chunks := []Chunk{
		makeChunk("/a.jsonl", "s1", "proj", "user", "user msg", 0),
		makeChunk("/a.jsonl", "s1", "proj", "assistant", "assistant msg", 1),
	}
	db.InsertChunks(chunks, [][]float32{emb, emb})

	results, err := db.Search(emb, 5, "", "user")
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(results) != 1 {
		t.Errorf("got %d results, want 1", len(results))
	}
	if results[0].Role != "user" {
		t.Errorf("Role = %q, want %q", results[0].Role, "user")
	}
}

func TestDeleteChunksByFile(t *testing.T) {
	db := tempDB(t)

	emb := make([]float32, 384)
	for i := range emb {
		emb[i] = 0.1
	}
	chunks := []Chunk{makeChunk("/test.jsonl", "s1", "proj", "user", "text", 0)}
	db.InsertChunks(chunks, [][]float32{emb})

	if err := db.DeleteChunksByFile("/test.jsonl"); err != nil {
		t.Fatalf("DeleteChunksByFile: %v", err)
	}

	results, err := db.Search(emb, 5, "", "")
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(results) != 0 {
		t.Errorf("got %d results after delete, want 0", len(results))
	}
}

func TestStats(t *testing.T) {
	db := tempDB(t)

	chunks, files, _, err := db.Stats()
	if err != nil {
		t.Fatalf("Stats: %v", err)
	}
	if chunks != 0 || files != 0 {
		t.Errorf("empty db: chunks=%d files=%d, want 0,0", chunks, files)
	}
}
```

- [ ] **Step 9: Implement chunk storage, vector search, delete, and stats**

Append to `internal/store/store.go`:

```go
func (db *DB) InsertChunks(chunks []Chunk, embeddings [][]float32) error {
	tx, err := db.conn.Begin()
	if err != nil {
		return err
	}
	defer tx.Rollback()

	chunkStmt, err := tx.Prepare(`INSERT INTO chunks
		(file_path, session_id, project, role, chunk_text, turn_index, created_at)
		VALUES (?, ?, ?, ?, ?, ?, ?)`)
	if err != nil {
		return err
	}
	defer chunkStmt.Close()

	vecStmt, err := tx.Prepare("INSERT INTO chunks_vec (id, embedding) VALUES (?, ?)")
	if err != nil {
		return err
	}
	defer vecStmt.Close()

	for i, c := range chunks {
		res, err := chunkStmt.Exec(c.FilePath, c.SessionID, c.Project, c.Role, c.Text, c.TurnIndex, c.CreatedAt)
		if err != nil {
			return err
		}
		id, err := res.LastInsertId()
		if err != nil {
			return err
		}
		if _, err := vecStmt.Exec(id, serializeF32(embeddings[i])); err != nil {
			return err
		}
	}
	return tx.Commit()
}

func (db *DB) DeleteChunksByFile(filePath string) error {
	if _, err := db.conn.Exec(
		"DELETE FROM chunks_vec WHERE id IN (SELECT id FROM chunks WHERE file_path = ?)",
		filePath); err != nil {
		return err
	}
	_, err := db.conn.Exec("DELETE FROM chunks WHERE file_path = ?", filePath)
	return err
}

func (db *DB) Search(queryEmbedding []float32, topK int, project, role string) ([]Result, error) {
	fetchK := topK
	if project != "" || role != "" {
		fetchK = topK * 5
	}

	rows, err := db.conn.Query(`
		SELECT v.id, v.distance, c.file_path, c.session_id, c.project,
		       c.role, c.chunk_text, c.turn_index, c.created_at
		FROM chunks_vec v
		INNER JOIN chunks c ON c.id = v.id
		WHERE v.embedding MATCH ?
		AND k = ?
		ORDER BY v.distance`,
		serializeF32(queryEmbedding), fetchK)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var results []Result
	for rows.Next() {
		var (
			id       int64
			distance float32
			r        Result
		)
		if err := rows.Scan(&id, &distance, &r.FilePath, &r.SessionID, &r.Project,
			&r.Role, &r.Text, &r.TurnIndex, &r.CreatedAt); err != nil {
			return nil, err
		}
		if project != "" && !hasPrefix(r.Project, project) {
			continue
		}
		if role != "" && r.Role != role {
			continue
		}
		r.Score = 1.0 - distance
		r.ResumeCommand = fmt.Sprintf("claude --resume %s", r.SessionID)
		results = append(results, r)
		if len(results) >= topK {
			break
		}
	}
	return results, rows.Err()
}

func hasPrefix(s, prefix string) bool {
	return len(s) >= len(prefix) && s[:len(prefix)] == prefix
}

func (db *DB) Stats() (chunks, files int, sizeKB int64, err error) {
	if err = db.conn.QueryRow("SELECT COUNT(*) FROM chunks").Scan(&chunks); err != nil {
		return
	}
	if err = db.conn.QueryRow("SELECT COUNT(*) FROM files").Scan(&files); err != nil {
		return
	}
	return
}
```

- [ ] **Step 10: Run all store tests**

```bash
CGO_ENABLED=1 go test ./internal/store/ -v -count=1
```

Expected: all tests PASS.

- [ ] **Step 11: Commit**

```bash
git add internal/store/
git commit -m "feat: add store package with SQLite + sqlite-vec persistence"
```

---

### Task 3: WordPiece Tokenizer

**Files:**
- Create: `internal/embedding/tokenizer.go`
- Create: `internal/embedding/tokenizer_test.go`

Pure Go WordPiece tokenizer that loads vocab.txt and produces token IDs for BERT model input.

- [ ] **Step 1: Write failing tests**

Create `internal/embedding/tokenizer_test.go`:

```go
package embedding

import (
	"os"
	"testing"
)

func loadTestVocab(t *testing.T) []byte {
	t.Helper()
	data, err := os.ReadFile("../../internal/assets/vocab.txt")
	if err != nil {
		t.Fatalf("read vocab.txt: %v", err)
	}
	return data
}

func TestNewTokenizer(t *testing.T) {
	vocab := loadTestVocab(t)
	tok := NewTokenizer(vocab)
	if len(tok.vocab) == 0 {
		t.Fatal("vocab is empty")
	}
	if _, ok := tok.vocab["[CLS]"]; !ok {
		t.Error("missing [CLS] token")
	}
	if _, ok := tok.vocab["[SEP]"]; !ok {
		t.Error("missing [SEP] token")
	}
	if _, ok := tok.vocab["[UNK]"]; !ok {
		t.Error("missing [UNK] token")
	}
}

func TestTokenizeSimple(t *testing.T) {
	tok := NewTokenizer(loadTestVocab(t))
	ids, mask, typeIDs := tok.Tokenize("hello world")

	if len(ids) < 3 {
		t.Fatalf("expected at least 3 tokens ([CLS] + words + [SEP]), got %d", len(ids))
	}
	if ids[0] != tok.vocab["[CLS]"] {
		t.Errorf("first token should be [CLS], got id %d", ids[0])
	}
	if ids[len(ids)-1] != tok.vocab["[SEP]"] {
		t.Errorf("last token should be [SEP], got id %d", ids[len(ids)-1])
	}
	if len(mask) != len(ids) {
		t.Errorf("mask length %d != ids length %d", len(mask), len(ids))
	}
	for i, m := range mask {
		if m != 1 {
			t.Errorf("mask[%d] = %d, want 1", i, m)
		}
	}
	if len(typeIDs) != len(ids) {
		t.Errorf("typeIDs length %d != ids length %d", len(typeIDs), len(ids))
	}
}

func TestCountTokens(t *testing.T) {
	tok := NewTokenizer(loadTestVocab(t))

	count := tok.CountTokens("hello world")
	if count < 2 {
		t.Errorf("expected at least 2 tokens for 'hello world', got %d", count)
	}

	count = tok.CountTokens("")
	if count != 0 {
		t.Errorf("expected 0 tokens for empty string, got %d", count)
	}
}

func TestTokenizeHandlesPunctuation(t *testing.T) {
	tok := NewTokenizer(loadTestVocab(t))
	ids, _, _ := tok.Tokenize("Hello, world!")
	if len(ids) < 5 {
		t.Errorf("expected at least 5 tokens (CLS + hello + , + world + ! + SEP), got %d", len(ids))
	}
}

func TestTokenizeUnknownWord(t *testing.T) {
	tok := NewTokenizer(loadTestVocab(t))
	ids, _, _ := tok.Tokenize("xyzzyplugh")
	unkID := tok.vocab["[UNK]"]
	hasUnk := false
	for _, id := range ids {
		if id == unkID {
			hasUnk = true
			break
		}
	}
	if !hasUnk {
		t.Error("expected [UNK] token for nonsense word")
	}
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
CGO_ENABLED=1 go test ./internal/embedding/ -v -run 'TestNewTokenizer|TestTokenize|TestCountTokens' -count=1
```

Expected: FAIL — `NewTokenizer` not defined.

- [ ] **Step 3: Implement the WordPiece tokenizer**

Create `internal/embedding/tokenizer.go`:

```go
package embedding

import (
	"strings"
	"unicode"

	"golang.org/x/text/unicode/norm"
)

type Tokenizer struct {
	vocab  map[string]int64
	maxLen int
}

func NewTokenizer(vocabData []byte) *Tokenizer {
	vocab := make(map[string]int64)
	for i, line := range strings.Split(string(vocabData), "\n") {
		line = strings.TrimRight(line, "\r")
		if line != "" {
			vocab[line] = int64(i)
		}
	}
	return &Tokenizer{vocab: vocab, maxLen: 512}
}

func (t *Tokenizer) Tokenize(text string) (inputIDs, attentionMask, tokenTypeIDs []int64) {
	tokens := t.tokenizeText(text)
	if len(tokens) > t.maxLen-2 {
		tokens = tokens[:t.maxLen-2]
	}

	ids := make([]int64, 0, len(tokens)+2)
	ids = append(ids, t.vocab["[CLS]"])
	for _, tok := range tokens {
		if id, ok := t.vocab[tok]; ok {
			ids = append(ids, id)
		} else {
			ids = append(ids, t.vocab["[UNK]"])
		}
	}
	ids = append(ids, t.vocab["[SEP]"])

	mask := make([]int64, len(ids))
	typeIDs := make([]int64, len(ids))
	for i := range mask {
		mask[i] = 1
	}
	return ids, mask, typeIDs
}

func (t *Tokenizer) CountTokens(text string) int {
	return len(t.tokenizeText(text))
}

func (t *Tokenizer) tokenizeText(text string) []string {
	text = strings.ToLower(text)
	text = stripAccents(text)
	words := splitOnWhitespaceAndPunctuation(text)

	var tokens []string
	for _, word := range words {
		tokens = append(tokens, t.wordPiece(word)...)
	}
	return tokens
}

func stripAccents(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	for _, r := range norm.NFD.String(s) {
		if !unicode.Is(unicode.Mn, r) {
			b.WriteRune(r)
		}
	}
	return b.String()
}

func splitOnWhitespaceAndPunctuation(text string) []string {
	var tokens []string
	var current strings.Builder
	for _, r := range text {
		if unicode.IsSpace(r) {
			if current.Len() > 0 {
				tokens = append(tokens, current.String())
				current.Reset()
			}
		} else if isPunct(r) {
			if current.Len() > 0 {
				tokens = append(tokens, current.String())
				current.Reset()
			}
			tokens = append(tokens, string(r))
		} else if unicode.IsControl(r) {
			continue
		} else {
			current.WriteRune(r)
		}
	}
	if current.Len() > 0 {
		tokens = append(tokens, current.String())
	}
	return tokens
}

func isPunct(r rune) bool {
	if (r >= '!' && r <= '/') || (r >= ':' && r <= '@') ||
		(r >= '[' && r <= '`') || (r >= '{' && r <= '~') {
		return true
	}
	return unicode.IsPunct(r) || unicode.IsSymbol(r)
}

func (t *Tokenizer) wordPiece(word string) []string {
	if _, ok := t.vocab[word]; ok {
		return []string{word}
	}

	runes := []rune(word)
	var tokens []string
	start := 0

	for start < len(runes) {
		end := len(runes)
		var matched string
		for end > start {
			substr := string(runes[start:end])
			if start > 0 {
				substr = "##" + substr
			}
			if _, ok := t.vocab[substr]; ok {
				matched = substr
				break
			}
			end--
		}
		if matched == "" {
			return []string{"[UNK]"}
		}
		tokens = append(tokens, matched)
		start = end
	}
	return tokens
}
```

- [ ] **Step 4: Get the text/unicode dependency**

```bash
go get golang.org/x/text/unicode/norm
go mod tidy
```

- [ ] **Step 5: Run tokenizer tests**

```bash
CGO_ENABLED=1 go test ./internal/embedding/ -v -run 'TestNewTokenizer|TestTokenize|TestCountTokens' -count=1
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add internal/embedding/tokenizer.go internal/embedding/tokenizer_test.go go.mod go.sum
git commit -m "feat: add pure Go WordPiece tokenizer"
```

---

### Task 4: ONNX Embedding Model

**Files:**
- Create: `internal/embedding/embedding.go`
- Create: `internal/embedding/embedding_test.go`

Wraps ONNX Runtime to load the embedded model, tokenize input, run inference, and produce 384-dimensional embedding vectors.

- [ ] **Step 1: Write failing tests**

Create `internal/embedding/embedding_test.go`:

```go
package embedding

import (
	"math"
	"testing"
)

func TestNewModel(t *testing.T) {
	model, err := New()
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer model.Close()

	if model.ModelHash() == "" {
		t.Error("ModelHash should not be empty")
	}
}

func TestEmbedSingleText(t *testing.T) {
	model, err := New()
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer model.Close()

	vecs, err := model.Embed([]string{"hello world"})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(vecs) != 1 {
		t.Fatalf("got %d vectors, want 1", len(vecs))
	}
	if len(vecs[0]) != 384 {
		t.Fatalf("vector dim = %d, want 384", len(vecs[0]))
	}
}

func TestEmbedBatch(t *testing.T) {
	model, err := New()
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer model.Close()

	texts := []string{"hello world", "how are you", "semantic search"}
	vecs, err := model.Embed(texts)
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(vecs) != 3 {
		t.Fatalf("got %d vectors, want 3", len(vecs))
	}
	for i, v := range vecs {
		if len(v) != 384 {
			t.Errorf("vecs[%d] dim = %d, want 384", i, len(v))
		}
	}
}

func TestEmbedSimilarity(t *testing.T) {
	model, err := New()
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer model.Close()

	vecs, err := model.Embed([]string{
		"How do I set up authentication?",
		"Configure JWT auth middleware",
		"The weather is nice today",
	})
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}

	simAB := cosine(vecs[0], vecs[1])
	simAC := cosine(vecs[0], vecs[2])

	if simAB <= simAC {
		t.Errorf("auth questions should be more similar (%.4f) than auth vs weather (%.4f)", simAB, simAC)
	}
}

func TestEmbedDeterministic(t *testing.T) {
	model, err := New()
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer model.Close()

	v1, _ := model.Embed([]string{"hello"})
	v2, _ := model.Embed([]string{"hello"})

	for i := range v1[0] {
		if v1[0][i] != v2[0][i] {
			t.Fatalf("embedding not deterministic at index %d: %f != %f", i, v1[0][i], v2[0][i])
		}
	}
}

func cosine(a, b []float32) float64 {
	var dot, normA, normB float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
CGO_ENABLED=1 go test ./internal/embedding/ -v -run 'TestNewModel|TestEmbed' -count=1
```

Expected: FAIL — `New` function not defined.

- [ ] **Step 3: Implement the embedding model**

Create `internal/embedding/embedding.go`:

```go
package embedding

import (
	"crypto/sha256"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"

	ort "github.com/yalue/onnxruntime_go"

	"github.com/mikeler216/cc-search/internal/assets"
)

var ortInitialized bool

type Model struct {
	modelPath string
	tokenizer *Tokenizer
	modelHash string
}

func onnxLibPath() string {
	switch runtime.GOOS {
	case "darwin":
		if runtime.GOARCH == "arm64" {
			return "/opt/homebrew/lib/libonnxruntime.dylib"
		}
		return "/usr/local/lib/libonnxruntime.dylib"
	default:
		return "/usr/local/lib/libonnxruntime.so"
	}
}

func New() (*Model, error) {
	if !ortInitialized {
		libPath := onnxLibPath()
		if p := os.Getenv("ONNX_LIB"); p != "" {
			libPath = p
		}
		ort.SetSharedLibraryPath(libPath)
		if err := ort.InitializeEnvironment(); err != nil {
			return nil, fmt.Errorf("init onnxruntime: %w", err)
		}
		ortInitialized = true
	}

	tmpDir, err := os.MkdirTemp("", "cc-search-model-*")
	if err != nil {
		return nil, err
	}
	modelPath := filepath.Join(tmpDir, "model.onnx")
	if err := os.WriteFile(modelPath, assets.ModelONNX, 0644); err != nil {
		return nil, err
	}

	hash := sha256.Sum256(assets.ModelONNX)

	return &Model{
		modelPath: modelPath,
		tokenizer: NewTokenizer(assets.VocabTxt),
		modelHash: fmt.Sprintf("%x", hash),
	}, nil
}

func (m *Model) Close() {
	os.RemoveAll(filepath.Dir(m.modelPath))
}

func (m *Model) ModelHash() string {
	return m.modelHash
}

func (m *Model) CountTokens(text string) int {
	return m.tokenizer.CountTokens(text)
}

func (m *Model) Embed(texts []string) ([][]float32, error) {
	results := make([][]float32, len(texts))
	batchSize := 64
	for start := 0; start < len(texts); start += batchSize {
		end := start + batchSize
		if end > len(texts) {
			end = len(texts)
		}
		batch := texts[start:end]
		vecs, err := m.embedBatch(batch)
		if err != nil {
			return nil, fmt.Errorf("embed batch [%d:%d]: %w", start, end, err)
		}
		copy(results[start:end], vecs)
	}
	return results, nil
}

func (m *Model) embedBatch(texts []string) ([][]float32, error) {
	batchSize := int64(len(texts))

	allIDs := make([][]int64, batchSize)
	allMasks := make([][]int64, batchSize)
	allTypes := make([][]int64, batchSize)
	maxLen := int64(0)

	for i, text := range texts {
		ids, mask, typeIDs := m.tokenizer.Tokenize(text)
		allIDs[i] = ids
		allMasks[i] = mask
		allTypes[i] = typeIDs
		if int64(len(ids)) > maxLen {
			maxLen = int64(len(ids))
		}
	}

	flatIDs := make([]int64, batchSize*maxLen)
	flatMask := make([]int64, batchSize*maxLen)
	flatTypes := make([]int64, batchSize*maxLen)

	for i := int64(0); i < batchSize; i++ {
		seqLen := int64(len(allIDs[i]))
		for j := int64(0); j < seqLen; j++ {
			flatIDs[i*maxLen+j] = allIDs[i][j]
			flatMask[i*maxLen+j] = allMasks[i][j]
			flatTypes[i*maxLen+j] = allTypes[i][j]
		}
	}

	shape := ort.NewShape(batchSize, maxLen)
	inputIDs, err := ort.NewTensor(shape, flatIDs)
	if err != nil {
		return nil, fmt.Errorf("create input_ids tensor: %w", err)
	}
	defer inputIDs.Destroy()

	attentionMask, err := ort.NewTensor(shape, flatMask)
	if err != nil {
		return nil, fmt.Errorf("create attention_mask tensor: %w", err)
	}
	defer attentionMask.Destroy()

	tokenTypeIDs, err := ort.NewTensor(shape, flatTypes)
	if err != nil {
		return nil, fmt.Errorf("create token_type_ids tensor: %w", err)
	}
	defer tokenTypeIDs.Destroy()

	outputShape := ort.NewShape(batchSize, maxLen, 384)
	output, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		return nil, fmt.Errorf("create output tensor: %w", err)
	}
	defer output.Destroy()

	session, err := ort.NewAdvancedSession(m.modelPath,
		[]string{"input_ids", "attention_mask", "token_type_ids"},
		[]string{"last_hidden_state"},
		[]ort.ArbitraryTensor{inputIDs, attentionMask, tokenTypeIDs},
		[]ort.ArbitraryTensor{output},
	)
	if err != nil {
		return nil, fmt.Errorf("create session: %w", err)
	}
	defer session.Destroy()

	if err := session.Run(); err != nil {
		return nil, fmt.Errorf("run session: %w", err)
	}

	raw := output.GetData()
	results := make([][]float32, batchSize)

	for i := int64(0); i < batchSize; i++ {
		seqLen := int64(len(allIDs[i]))
		vec := make([]float32, 384)
		maskSum := float32(0)

		for j := int64(0); j < seqLen; j++ {
			maskVal := float32(flatMask[i*maxLen+j])
			maskSum += maskVal
			for k := int64(0); k < 384; k++ {
				vec[k] += raw[i*maxLen*384+j*384+k] * maskVal
			}
		}

		if maskSum > 0 {
			for k := range vec {
				vec[k] /= maskSum
			}
		}

		norm := float32(0)
		for _, v := range vec {
			norm += v * v
		}
		norm = float32(math.Sqrt(float64(norm)))
		if norm > 0 {
			for k := range vec {
				vec[k] /= norm
			}
		}

		results[i] = vec
	}

	return results, nil
}
```

- [ ] **Step 4: Run embedding tests**

```bash
CGO_ENABLED=1 go test ./internal/embedding/ -v -run 'TestNewModel|TestEmbed' -count=1 -timeout 120s
```

Expected: all tests PASS. The first run may take a few seconds for ONNX Runtime initialization.

- [ ] **Step 5: Commit**

```bash
git add internal/embedding/embedding.go internal/embedding/embedding_test.go
git commit -m "feat: add ONNX embedding model with mean pooling and L2 normalization"
```

---

### Task 5: Ingest Pipeline — Parsing & Chunking

**Files:**
- Create: `internal/ingest/parse.go`
- Create: `internal/ingest/chunk.go`
- Create: `internal/ingest/ingest_test.go`

JSONL parsing and token-aware chunking — the data transformation layer of the ingest pipeline.

- [ ] **Step 1: Write failing tests for JSONL parsing**

Create `internal/ingest/ingest_test.go`:

```go
package ingest

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func writeJSONL(t *testing.T, dir, filename string, entries []map[string]any) string {
	t.Helper()
	path := filepath.Join(dir, filename)
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	for _, e := range entries {
		enc.Encode(e)
	}
	return path
}

func TestExtractTextString(t *testing.T) {
	text := extractText(map[string]any{
		"role":    "user",
		"content": "hello world",
	})
	if text != "hello world" {
		t.Errorf("got %q, want %q", text, "hello world")
	}
}

func TestExtractTextList(t *testing.T) {
	text := extractText(map[string]any{
		"role": "assistant",
		"content": []any{
			map[string]any{"type": "thinking", "thinking": "let me think"},
			map[string]any{"type": "text", "text": "Here is the answer."},
			map[string]any{"type": "tool_use", "name": "Bash"},
		},
	})
	if text != "Here is the answer." {
		t.Errorf("got %q, want %q", text, "Here is the answer.")
	}
}

func TestExtractTextToolResult(t *testing.T) {
	text := extractText(map[string]any{
		"role": "user",
		"content": []any{
			map[string]any{"type": "tool_result", "content": "some output"},
		},
	})
	if text != "" {
		t.Errorf("got %q, want empty", text)
	}
}

func TestParseJSONLFile(t *testing.T) {
	dir := t.TempDir()
	entries := []map[string]any{
		{"type": "permission-mode", "permissionMode": "default", "sessionId": "sess-1"},
		{
			"type":      "user",
			"message":   map[string]any{"role": "user", "content": "How do I fix this?"},
			"sessionId": "sess-1",
			"timestamp": 1000.0,
		},
		{
			"type": "assistant",
			"message": map[string]any{
				"role": "assistant",
				"content": []any{
					map[string]any{"type": "text", "text": "Check the logs."},
				},
			},
			"sessionId": "sess-1",
			"timestamp": 1001.0,
		},
	}
	path := writeJSONL(t, dir, "sess-1.jsonl", entries)

	turns := parseJSONLFile(path)
	if len(turns) != 2 {
		t.Fatalf("got %d turns, want 2", len(turns))
	}
	if turns[0].Role != "user" || turns[0].Text != "How do I fix this?" {
		t.Errorf("turn 0: role=%q text=%q", turns[0].Role, turns[0].Text)
	}
	if turns[1].Role != "assistant" || turns[1].Text != "Check the logs." {
		t.Errorf("turn 1: role=%q text=%q", turns[1].Role, turns[1].Text)
	}
	if turns[0].SessionID != "sess-1" {
		t.Errorf("session_id = %q, want %q", turns[0].SessionID, "sess-1")
	}
}

func TestProjectNameFromPath(t *testing.T) {
	tests := []struct {
		path string
		want string
	}{
		{"/home/user/.claude/projects/-Users-test-myproj/sess.jsonl", "-Users-test-myproj"},
		{"/no/projects/here.jsonl", "unknown"},
	}
	for _, tt := range tests {
		got := projectNameFromPath(tt.path)
		if got != tt.want {
			t.Errorf("projectNameFromPath(%q) = %q, want %q", tt.path, got, tt.want)
		}
	}
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
CGO_ENABLED=1 go test ./internal/ingest/ -v -run 'TestExtract|TestParseJSONL|TestProjectName' -count=1
```

Expected: FAIL — package doesn't exist yet.

- [ ] **Step 3: Implement JSONL parsing**

Create `internal/ingest/parse.go`:

```go
package ingest

import (
	"bufio"
	"encoding/json"
	"os"
	"strings"
)

type turn struct {
	Role      string
	Text      string
	SessionID string
	Timestamp float64
	TurnIndex int
}

func extractText(message map[string]any) string {
	content, ok := message["content"]
	if !ok {
		return ""
	}

	switch c := content.(type) {
	case string:
		return strings.TrimSpace(c)
	case []any:
		var parts []string
		for _, block := range c {
			m, ok := block.(map[string]any)
			if !ok {
				continue
			}
			if m["type"] == "text" {
				if text, ok := m["text"].(string); ok {
					parts = append(parts, text)
				}
			}
		}
		return strings.TrimSpace(strings.Join(parts, " "))
	}
	return ""
}

func projectNameFromPath(filePath string) string {
	parts := strings.Split(filePath, "/projects/")
	if len(parts) < 2 {
		return "unknown"
	}
	segments := strings.SplitN(parts[1], "/", 2)
	return segments[0]
}

func parseJSONLFile(filePath string) []turn {
	f, err := os.Open(filePath)
	if err != nil {
		return nil
	}
	defer f.Close()

	var turns []turn
	turnIndex := 0
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 0, 1024*1024), 10*1024*1024)

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var entry map[string]any
		if err := json.Unmarshal([]byte(line), &entry); err != nil {
			continue
		}
		entryType, _ := entry["type"].(string)
		if entryType != "user" && entryType != "assistant" {
			continue
		}
		message, _ := entry["message"].(map[string]any)
		if message == nil {
			continue
		}
		text := extractText(message)
		if text == "" {
			continue
		}
		sessionID, _ := entry["sessionId"].(string)
		timestamp, _ := entry["timestamp"].(float64)

		turns = append(turns, turn{
			Role:      entryType,
			Text:      text,
			SessionID: sessionID,
			Timestamp: timestamp,
			TurnIndex: turnIndex,
		})
		turnIndex++
	}
	return turns
}
```

- [ ] **Step 4: Run parsing tests**

```bash
CGO_ENABLED=1 go test ./internal/ingest/ -v -run 'TestExtract|TestParseJSONL|TestProjectName' -count=1
```

Expected: PASS

- [ ] **Step 5: Write failing tests for token-aware chunking**

Append to `internal/ingest/ingest_test.go`:

```go
type fakeCounter struct{}

func (f fakeCounter) CountTokens(text string) int {
	return len(strings.Fields(text))
}

func TestChunkShortText(t *testing.T) {
	chunks := chunkText("Hello, how are you?", fakeCounter{}, 400, 100)
	if len(chunks) != 1 {
		t.Fatalf("got %d chunks, want 1", len(chunks))
	}
	if chunks[0] != "Hello, how are you?" {
		t.Errorf("got %q", chunks[0])
	}
}

func TestChunkEmpty(t *testing.T) {
	chunks := chunkText("", fakeCounter{}, 400, 100)
	if len(chunks) != 0 {
		t.Errorf("got %d chunks, want 0", len(chunks))
	}
}

func TestChunkWhitespace(t *testing.T) {
	chunks := chunkText("   \n\n  ", fakeCounter{}, 400, 100)
	if len(chunks) != 0 {
		t.Errorf("got %d chunks, want 0", len(chunks))
	}
}

func TestChunkLongText(t *testing.T) {
	words := make([]string, 600)
	for i := range words {
		words[i] = "word"
	}
	text := strings.Join(words, " ")
	chunks := chunkText(text, fakeCounter{}, 400, 100)

	if len(chunks) < 2 {
		t.Fatalf("got %d chunks, want >= 2", len(chunks))
	}
	for _, c := range chunks {
		wordCount := len(strings.Fields(c))
		if wordCount > 420 {
			t.Errorf("chunk has %d words, max should be ~400", wordCount)
		}
	}
}

func TestChunkOverlap(t *testing.T) {
	words := make([]string, 800)
	for i := range words {
		words[i] = fmt.Sprintf("word%d", i)
	}
	text := strings.Join(words, " ")
	chunks := chunkText(text, fakeCounter{}, 400, 100)

	if len(chunks) < 2 {
		t.Fatalf("need at least 2 chunks")
	}
	firstWords := toSet(strings.Fields(chunks[0]))
	secondWords := toSet(strings.Fields(chunks[1]))
	overlap := 0
	for w := range firstWords {
		if secondWords[w] {
			overlap++
		}
	}
	if overlap == 0 {
		t.Error("expected overlap between consecutive chunks")
	}
}

func toSet(words []string) map[string]bool {
	s := make(map[string]bool, len(words))
	for _, w := range words {
		s[w] = true
	}
	return s
}
```

- [ ] **Step 6: Implement token-aware chunking**

Create `internal/ingest/chunk.go`:

```go
package ingest

import "strings"

type tokenCounter interface {
	CountTokens(text string) int
}

func chunkText(text string, counter tokenCounter, maxTokens, overlapTokens int) []string {
	text = strings.TrimSpace(text)
	if text == "" {
		return nil
	}

	if counter.CountTokens(text) <= maxTokens {
		return []string{text}
	}

	words := strings.Fields(text)
	var chunks []string
	start := 0

	for start < len(words) {
		end := start + maxTokens
		if end > len(words) {
			end = len(words)
		}

		for end > start+1 {
			candidate := strings.Join(words[start:end], " ")
			if counter.CountTokens(candidate) <= maxTokens {
				break
			}
			end--
		}

		chunk := strings.Join(words[start:end], " ")
		chunks = append(chunks, chunk)

		if end >= len(words) {
			break
		}
		step := end - start - overlapTokens
		if step < 1 {
			step = 1
		}
		start += step
	}

	return chunks
}
```

- [ ] **Step 7: Add the `fmt` and `strings` imports to the test file header**

The test file uses `fmt` (in `TestChunkOverlap`) and `strings` (in `TestChunkShortText`). Make sure the import block at the top of `ingest_test.go` is:

```go
import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)
```

- [ ] **Step 8: Run all ingest tests**

```bash
CGO_ENABLED=1 go test ./internal/ingest/ -v -count=1
```

Expected: all tests PASS.

- [ ] **Step 9: Commit**

```bash
git add internal/ingest/
git commit -m "feat: add JSONL parsing and token-aware chunking"
```

---

### Task 6: Ingest Pipeline — Orchestration

**Files:**
- Create: `internal/ingest/ingest.go`
- Modify: `internal/ingest/ingest_test.go` (add integration tests)

The pipeline orchestrator: find files, filter by mtime, parse → chunk → embed → store.

- [ ] **Step 1: Write failing integration test**

Append to `internal/ingest/ingest_test.go`:

```go
func TestRunIndexesFiles(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test")
	}

	dir := t.TempDir()
	claudeDir := dir
	projectsDir := filepath.Join(claudeDir, "projects", "-Users-test-myproject")
	os.MkdirAll(projectsDir, 0755)

	entries := []map[string]any{
		{
			"type":      "user",
			"message":   map[string]any{"role": "user", "content": "What is semantic search?"},
			"sessionId": "sess-1",
			"timestamp": 1000.0,
		},
		{
			"type": "assistant",
			"message": map[string]any{
				"role":    "assistant",
				"content": []any{map[string]any{"type": "text", "text": "Semantic search uses embeddings."}},
			},
			"sessionId": "sess-1",
			"timestamp": 1001.0,
		},
	}
	writeJSONL(t, projectsDir, "sess-1.jsonl", entries)

	dbPath := filepath.Join(dir, "test.db")

	emb, err := newTestModel(t)
	if err != nil {
		t.Fatalf("new model: %v", err)
	}
	defer emb.Close()

	chunks, files, err := Run(dbPath, emb, claudeDir, false)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if files != 1 {
		t.Errorf("files = %d, want 1", files)
	}
	if chunks < 2 {
		t.Errorf("chunks = %d, want >= 2", chunks)
	}
}

func TestRunIncrementalSkipsUnchanged(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test")
	}

	dir := t.TempDir()
	claudeDir := dir
	projectsDir := filepath.Join(claudeDir, "projects", "-Users-test-proj")
	os.MkdirAll(projectsDir, 0755)

	entries := []map[string]any{
		{
			"type":      "user",
			"message":   map[string]any{"role": "user", "content": "Hello"},
			"sessionId": "s1",
			"timestamp": 1000.0,
		},
	}
	writeJSONL(t, projectsDir, "s1.jsonl", entries)

	dbPath := filepath.Join(dir, "test.db")

	emb, err := newTestModel(t)
	if err != nil {
		t.Fatalf("new model: %v", err)
	}
	defer emb.Close()

	Run(dbPath, emb, claudeDir, false)
	chunks2, _, err := Run(dbPath, emb, claudeDir, false)
	if err != nil {
		t.Fatalf("second Run: %v", err)
	}
	if chunks2 != 0 {
		t.Errorf("second run indexed %d chunks, want 0 (incremental skip)", chunks2)
	}
}
```

We need a helper to create the embedding model in tests. Add at the top of the test file (within the existing import block, add the embedding import):

```go
import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/mikeler216/cc-search/internal/embedding"
)

func newTestModel(t *testing.T) (*embedding.Model, error) {
	t.Helper()
	return embedding.New()
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
CGO_ENABLED=1 go test ./internal/ingest/ -v -run 'TestRun' -count=1 -timeout 120s
```

Expected: FAIL — `Run` not defined.

- [ ] **Step 3: Implement the ingest orchestrator**

Create `internal/ingest/ingest.go`:

```go
package ingest

import (
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/mikeler216/cc-search/internal/embedding"
	"github.com/mikeler216/cc-search/internal/store"
)

func Run(dbPath string, model *embedding.Model, claudeDir string, full bool) (indexedChunks, indexedFiles int, err error) {
	db, err := store.Open(dbPath)
	if err != nil {
		return 0, 0, err
	}
	defer db.Close()

	match, err := db.CheckVersion(model.ModelHash())
	if err != nil || !match {
		if err := db.DropAll(); err != nil {
			return 0, 0, err
		}
		full = true
	}
	if err := db.InitMeta(model.ModelHash()); err != nil {
		return 0, 0, err
	}

	files := findJSONLFiles(claudeDir)
	fileSet := make(map[string]bool, len(files))
	for _, f := range files {
		fileSet[f] = true
	}

	existing, err := db.AllFiles()
	if err != nil {
		return 0, 0, err
	}
	for path := range existing {
		if !fileSet[path] {
			db.DeleteChunksByFile(path)
		}
	}

	for _, filePath := range files {
		mtime := fileMtime(filePath)
		if !full {
			if rec, ok := existing[filePath]; ok && rec.LastModified >= mtime {
				continue
			}
		}

		n, err := indexFile(db, model, filePath, mtime)
		if err != nil {
			return indexedChunks, indexedFiles, err
		}
		indexedChunks += n
		indexedFiles++
	}

	return indexedChunks, indexedFiles, nil
}

func findJSONLFiles(claudeDir string) []string {
	projectsDir := filepath.Join(claudeDir, "projects")
	if _, err := os.Stat(projectsDir); os.IsNotExist(err) {
		return nil
	}

	var files []string
	filepath.Walk(projectsDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return nil
		}
		if info.IsDir() && info.Name() == "subagents" {
			return filepath.SkipDir
		}
		if !info.IsDir() && strings.HasSuffix(path, ".jsonl") {
			files = append(files, path)
		}
		return nil
	})
	return files
}

func indexFile(db *store.DB, model *embedding.Model, filePath string, mtime float64) (int, error) {
	db.DeleteChunksByFile(filePath)

	turns := parseJSONLFile(filePath)
	if len(turns) == 0 {
		db.UpsertFile(filePath, mtime, float64(time.Now().Unix()))
		return 0, nil
	}

	project := projectNameFromPath(filePath)

	var allChunks []store.Chunk
	var allTexts []string

	for _, t := range turns {
		textChunks := chunkText(t.Text, model, 400, 100)
		for _, chunk := range textChunks {
			allChunks = append(allChunks, store.Chunk{
				FilePath:  filePath,
				SessionID: t.SessionID,
				Project:   project,
				Role:      t.Role,
				Text:      chunk,
				TurnIndex: t.TurnIndex,
				CreatedAt: t.Timestamp,
			})
			allTexts = append(allTexts, chunk)
		}
	}

	if len(allTexts) == 0 {
		db.UpsertFile(filePath, mtime, float64(time.Now().Unix()))
		return 0, nil
	}

	embeddings, err := model.Embed(allTexts)
	if err != nil {
		return 0, err
	}

	if err := db.InsertChunks(allChunks, embeddings); err != nil {
		return 0, err
	}

	db.UpsertFile(filePath, mtime, float64(time.Now().Unix()))
	return len(allChunks), nil
}

func fileMtime(path string) float64 {
	info, err := os.Stat(path)
	if err != nil {
		return 0
	}
	return float64(info.ModTime().Unix())
}
```

- [ ] **Step 4: Run integration tests**

```bash
CGO_ENABLED=1 go test ./internal/ingest/ -v -run 'TestRun' -count=1 -timeout 120s
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add internal/ingest/ingest.go
git commit -m "feat: add ingest pipeline orchestrator with incremental indexing"
```

---

### Task 7: Search Package

**Files:**
- Create: `internal/search/search.go`
- Create: `internal/search/search_test.go`

Query embedding, vector search, and result formatting.

- [ ] **Step 1: Write failing tests**

Create `internal/search/search_test.go`:

```go
package search

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/mikeler216/cc-search/internal/embedding"
	"github.com/mikeler216/cc-search/internal/store"
)

func seedDB(t *testing.T, db *store.DB, model *embedding.Model) {
	t.Helper()
	texts := []struct {
		text string
		role string
	}{
		{"How do I set up JWT auth middleware?", "user"},
		{"Here is how to configure JWT validation.", "assistant"},
		{"Fix the database migration bug", "user"},
		{"The migration failed because of a missing column.", "assistant"},
		{"How does React state management work?", "user"},
	}

	chunks := make([]store.Chunk, len(texts))
	rawTexts := make([]string, len(texts))
	for i, tt := range texts {
		chunks[i] = store.Chunk{
			FilePath:  "/test.jsonl",
			SessionID: "sess-1",
			Project:   "-Users-test-myproject",
			Role:      tt.role,
			Text:      tt.text,
			TurnIndex: i,
			CreatedAt: 1000.0 + float64(i),
		}
		rawTexts[i] = tt.text
	}

	embeddings, err := model.Embed(rawTexts)
	if err != nil {
		t.Fatal(err)
	}
	if err := db.InsertChunks(chunks, embeddings); err != nil {
		t.Fatal(err)
	}
}

func setup(t *testing.T) (*store.DB, *embedding.Model) {
	t.Helper()
	dir := t.TempDir()
	db, err := store.Open(filepath.Join(dir, "test.db"))
	if err != nil {
		t.Fatal(err)
	}
	model, err := embedding.New()
	if err != nil {
		t.Fatal(err)
	}
	db.InitMeta(model.ModelHash())
	t.Cleanup(func() {
		db.Close()
		model.Close()
	})
	return db, model
}

func TestSearchReturnsRelevantResults(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test")
	}
	db, model := setup(t)
	seedDB(t, db, model)

	results, err := Run(db, model, "authentication setup", 5, "", "")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("expected results")
	}
	top := results[0].Text
	if !(contains(top, "JWT") || contains(top, "auth")) {
		t.Errorf("top result %q doesn't mention JWT or auth", top)
	}
}

func TestSearchTopK(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test")
	}
	db, model := setup(t)
	seedDB(t, db, model)

	results, err := Run(db, model, "anything", 2, "", "")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if len(results) > 2 {
		t.Errorf("got %d results, want <= 2", len(results))
	}
}

func TestSearchEmptyDB(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test")
	}
	db, model := setup(t)

	results, err := Run(db, model, "hello", 5, "", "")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if len(results) != 0 {
		t.Errorf("got %d results, want 0", len(results))
	}
}

func TestSearchResumeCommand(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping integration test")
	}
	db, model := setup(t)
	seedDB(t, db, model)

	results, err := Run(db, model, "database migration", 5, "", "")
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("expected results")
	}
	if results[0].ResumeCommand != "claude --resume sess-1" {
		t.Errorf("ResumeCommand = %q", results[0].ResumeCommand)
	}
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(s) > 0 && containsStr(s, substr))
}

func containsStr(s, sub string) bool {
	for i := 0; i <= len(s)-len(sub); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
CGO_ENABLED=1 go test ./internal/search/ -v -run 'TestSearch' -count=1 -timeout 120s
```

Expected: FAIL — `Run` not defined.

- [ ] **Step 3: Implement the search package**

Create `internal/search/search.go`:

```go
package search

import (
	"github.com/mikeler216/cc-search/internal/embedding"
	"github.com/mikeler216/cc-search/internal/store"
)

func Run(db *store.DB, model *embedding.Model, query string, topK int, project, role string) ([]store.Result, error) {
	vecs, err := model.Embed([]string{query})
	if err != nil {
		return nil, err
	}
	return db.Search(vecs[0], topK, project, role)
}
```

- [ ] **Step 4: Run search tests**

```bash
CGO_ENABLED=1 go test ./internal/search/ -v -run 'TestSearch' -count=1 -timeout 120s
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add internal/search/
git commit -m "feat: add search package with query embedding and vector search"
```

---

### Task 8: CLI Commands

**Files:**
- Modify: `cmd/cc-search/main.go`

All cobra commands: `index`, `query`, `status`, `update`.

- [ ] **Step 1: Implement the full CLI**

Replace `cmd/cc-search/main.go`:

```go
package main

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/fsnotify/fsnotify"
	"github.com/spf13/cobra"

	"github.com/mikeler216/cc-search/internal/embedding"
	"github.com/mikeler216/cc-search/internal/ingest"
	"github.com/mikeler216/cc-search/internal/search"
	"github.com/mikeler216/cc-search/internal/store"
)

var (
	defaultClaudeDir = filepath.Join(os.Getenv("HOME"), ".claude")
	defaultDBPath    = filepath.Join(defaultClaudeDir, "search-index", "conversations.db")
)

func main() {
	if err := rootCmd().Execute(); err != nil {
		os.Exit(1)
	}
}

func rootCmd() *cobra.Command {
	root := &cobra.Command{
		Use:   "cc-search",
		Short: "Semantic search over Claude Code conversation history",
	}
	root.AddCommand(indexCmd(), queryCmd(), statusCmd(), updateCmd())
	return root
}

func indexCmd() *cobra.Command {
	var dbPath, claudeDir string
	var full, watch bool

	cmd := &cobra.Command{
		Use:   "index",
		Short: "Build or update the search index",
		RunE: func(cmd *cobra.Command, args []string) error {
			model, err := embedding.New()
			if err != nil {
				return err
			}
			defer model.Close()

			if watch {
				return runWatch(dbPath, model, claudeDir)
			}

			chunks, files, err := ingest.Run(dbPath, model, claudeDir, full)
			if err != nil {
				return err
			}

			db, err := store.Open(dbPath)
			if err != nil {
				return err
			}
			defer db.Close()
			totalChunks, totalFiles, _, _ := db.Stats()
			if chunks > 0 || files > 0 {
				fmt.Printf("Indexed %d chunks from %d files.\n", totalChunks, totalFiles)
			} else {
				fmt.Printf("Index up to date. %d chunks from %d files.\n", totalChunks, totalFiles)
			}
			return nil
		},
	}
	cmd.Flags().StringVar(&dbPath, "db-path", defaultDBPath, "Path to the search database")
	cmd.Flags().StringVar(&claudeDir, "claude-dir", defaultClaudeDir, "Path to ~/.claude directory")
	cmd.Flags().BoolVar(&full, "full", false, "Force full reindex")
	cmd.Flags().BoolVar(&watch, "watch", false, "Watch for changes and reindex")
	return cmd
}

func runWatch(dbPath string, model *embedding.Model, claudeDir string) error {
	fmt.Println("Watching for changes... (Ctrl+C to stop)")

	projectsDir := filepath.Join(claudeDir, "projects")
	watcher, err := fsnotify.NewWatcher()
	if err != nil {
		return err
	}
	defer watcher.Close()

	if err := watcher.Add(projectsDir); err != nil {
		fmt.Fprintf(os.Stderr, "fsnotify watch failed, falling back to polling: %v\n", err)
		for {
			ingest.Run(dbPath, model, claudeDir, false)
			time.Sleep(30 * time.Second)
		}
	}

	filepath.Walk(projectsDir, func(path string, info os.FileInfo, err error) error {
		if err == nil && info.IsDir() {
			watcher.Add(path)
		}
		return nil
	})

	ingest.Run(dbPath, model, claudeDir, false)

	debounce := time.NewTimer(0)
	<-debounce.C

	for {
		select {
		case event, ok := <-watcher.Events:
			if !ok {
				return nil
			}
			if strings.HasSuffix(event.Name, ".jsonl") {
				debounce.Reset(2 * time.Second)
			}
		case <-debounce.C:
			ingest.Run(dbPath, model, claudeDir, false)
		case err, ok := <-watcher.Errors:
			if !ok {
				return nil
			}
			fmt.Fprintf(os.Stderr, "watch error: %v\n", err)
		}
	}
}

func cwdToProject(cwd string) string {
	return "-" + strings.ReplaceAll(strings.TrimLeft(cwd, "/"), "/", "-")
}

func displayProject(rawName string) string {
	segments := strings.Split(strings.TrimLeft(rawName, "-"), "-")
	resolved := "/"
	i := 0
	for i < len(segments) {
		matched := false
		for j := len(segments); j > i; j-- {
			candidate := strings.Join(segments[i:j], "-")
			testPath := filepath.Join(resolved, candidate)
			if info, err := os.Stat(testPath); (err == nil && info.IsDir()) || j == i+1 {
				resolved = testPath
				i = j
				matched = true
				break
			}
		}
		if !matched {
			break
		}
	}
	home := os.Getenv("HOME")
	if strings.HasPrefix(resolved, home+"/") {
		return resolved[len(home)+1:]
	}
	return strings.TrimLeft(resolved, "/")
}

func queryCmd() *cobra.Command {
	var dbPath, claudeDir, project, role string
	var top int
	var searchAll bool

	cmd := &cobra.Command{
		Use:   "query [search terms...]",
		Short: "Semantic search over conversations",
		Args:  cobra.MinimumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			queryText := strings.Join(args, " ")
			if queryText == "" {
				fmt.Println("No query provided.")
				return nil
			}

			if !searchAll && project == "" {
				cwd, _ := os.Getwd()
				project = cwdToProject(cwd)
			}

			model, err := embedding.New()
			if err != nil {
				return err
			}
			defer model.Close()

			db, err := store.Open(dbPath)
			if err != nil {
				return err
			}
			defer db.Close()

			match, err := db.CheckVersion(model.ModelHash())
			if err != nil || !match {
				fmt.Fprintln(os.Stderr, "Index out of date. Run `cc-search index` to rebuild.")
				os.Exit(1)
			}

			// Run incremental index before searching (picks up new files)
			ingest.Run(dbPath, model, claudeDir, false)

			results, err := search.Run(db, model, queryText, top, project, role)
			if err != nil {
				return err
			}

			if len(results) == 0 {
				fmt.Println("No results found.")
				return nil
			}

			for i, r := range results {
				fmt.Printf("── Result %d (score: %.2f) %s\n", i+1, r.Score, strings.Repeat("─", 30))
				fmt.Printf("Project: %s\n", displayProject(r.Project))
				fmt.Printf("Session: %s\n", r.SessionID)
				fmt.Printf("Role:    %s\n", r.Role)
				fmt.Printf("Turn:    %d\n", r.TurnIndex)
				fmt.Println()
				text := r.Text
				if len(text) > 300 {
					text = text[:300] + "..."
				}
				for _, line := range strings.Split(text, "\n") {
					fmt.Printf("  %s\n", line)
				}
				fmt.Println()
				fmt.Printf("  → %s\n", r.ResumeCommand)
				fmt.Println()
			}
			return nil
		},
	}
	cmd.Flags().StringVar(&dbPath, "db-path", defaultDBPath, "Path to the search database")
	cmd.Flags().StringVar(&claudeDir, "claude-dir", defaultClaudeDir, "Path to ~/.claude directory")
	cmd.Flags().IntVar(&top, "top", 5, "Number of results")
	cmd.Flags().BoolVar(&searchAll, "all", false, "Search all projects")
	cmd.Flags().StringVar(&project, "project", "", "Filter by project path")
	cmd.Flags().StringVar(&role, "role", "", "Filter by role (user|assistant)")
	return cmd
}

func statusCmd() *cobra.Command {
	var dbPath string

	cmd := &cobra.Command{
		Use:   "status",
		Short: "Show index statistics",
		RunE: func(cmd *cobra.Command, args []string) error {
			db, err := store.Open(dbPath)
			if err != nil {
				return err
			}
			defer db.Close()

			model, err := embedding.New()
			if err != nil {
				return err
			}
			defer model.Close()

			match, _ := db.CheckVersion(model.ModelHash())
			if !match {
				fmt.Fprintln(os.Stderr, "Index out of date. Run `cc-search index` to rebuild.")
				os.Exit(1)
			}

			chunks, files, _, err := db.Stats()
			if err != nil {
				return err
			}

			dbSize := int64(0)
			if info, err := os.Stat(dbPath); err == nil {
				dbSize = info.Size()
			}
			fmt.Printf("Chunks:  %d\n", chunks)
			fmt.Printf("Files:   %d\n", files)
			fmt.Printf("DB size: %.1f KB\n", float64(dbSize)/1024)
			return nil
		},
	}
	cmd.Flags().StringVar(&dbPath, "db-path", defaultDBPath, "Path to the search database")
	return cmd
}

func updateCmd() *cobra.Command {
	return &cobra.Command{
		Use:   "update",
		Short: "Update cc-search to the latest version",
		RunE: func(cmd *cobra.Command, args []string) error {
			fmt.Println("Checking for updates...")

			goos := runtime.GOOS
			goarch := runtime.GOARCH
			binaryName := fmt.Sprintf("cc-search-%s-%s", goos, goarch)
			url := fmt.Sprintf("https://github.com/mikeler216/cc-search/releases/latest/download/%s", binaryName)

			resp, err := http.Get(url)
			if err != nil {
				return fmt.Errorf("download failed: %w", err)
			}
			defer resp.Body.Close()

			if resp.StatusCode != 200 {
				return fmt.Errorf("download failed: HTTP %d", resp.StatusCode)
			}

			self, err := os.Executable()
			if err != nil {
				return err
			}
			self, err = filepath.EvalSymlinks(self)
			if err != nil {
				return err
			}

			tmpPath := self + ".new"
			f, err := os.OpenFile(tmpPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0755)
			if err != nil {
				return err
			}
			if _, err := io.Copy(f, resp.Body); err != nil {
				f.Close()
				os.Remove(tmpPath)
				return err
			}
			f.Close()

			if err := os.Rename(tmpPath, self); err != nil {
				os.Remove(tmpPath)
				return fmt.Errorf("replace binary: %w", err)
			}

			fmt.Println("Updated successfully. Reindexing...")
			reindex := exec.Command(self, "index", "--full")
			reindex.Stdout = os.Stdout
			reindex.Stderr = os.Stderr
			return reindex.Run()
		},
	}
}
```

- [ ] **Step 2: Verify it compiles**

```bash
CGO_ENABLED=1 go build -o cc-search ./cmd/cc-search
```

Expected: binary produced.

- [ ] **Step 3: Smoke test the CLI**

```bash
./cc-search --help
./cc-search status --db-path /tmp/cc-search-test.db
```

Expected: help output shows all commands. Status shows 0 chunks/files for empty db.

- [ ] **Step 4: Commit**

```bash
git add cmd/cc-search/main.go
git commit -m "feat: add CLI with index, query, status, and update commands"
```

---

### Task 9: Plugin & Script Updates

**Files:**
- Modify: `scripts/install.sh`
- Modify: `commands/search-history.md`
- Modify: `skills/search-history/SKILL.md`
- Modify: `.claude-plugin/plugin.json`

Update all plugin integration files to work with the Go binary instead of Python/uv.

- [ ] **Step 1: Rewrite install.sh for binary distribution**

Replace `scripts/install.sh`:

```bash
#!/usr/bin/env bash
set -e

echo "cc-search installer"
echo "==================="
echo ""
echo "This will install:"
echo "  1. cc-search binary (semantic search over Claude Code conversations)"
echo "  2. Build the search index from your conversation history"
echo ""
read -p "Continue? [Y/n] " answer
if [[ "$answer" =~ ^[Nn] ]]; then
  echo "Cancelled."
  exit 0
fi

echo ""

OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m)"
case "$ARCH" in
  x86_64) ARCH="amd64" ;;
  aarch64|arm64) ARCH="arm64" ;;
esac

BINARY="cc-search-${OS}-${ARCH}"
URL="https://github.com/mikeler216/cc-search/releases/latest/download/${BINARY}"

INSTALL_DIR="$HOME/.local/bin"
mkdir -p "$INSTALL_DIR"

echo "Downloading cc-search for ${OS}/${ARCH}..."
if command -v curl &>/dev/null; then
  curl -L -o "${INSTALL_DIR}/cc-search" "$URL"
elif command -v wget &>/dev/null; then
  wget -O "${INSTALL_DIR}/cc-search" "$URL"
else
  echo "Error: curl or wget required"
  exit 1
fi

chmod +x "${INSTALL_DIR}/cc-search"
echo "✓ cc-search installed to ${INSTALL_DIR}/cc-search"

if ! echo "$PATH" | grep -q "$INSTALL_DIR"; then
  echo ""
  echo "Add to your PATH: export PATH=\"$INSTALL_DIR:\$PATH\""
fi

echo ""
echo "Building search index (this may take a minute on first run)..."
"${INSTALL_DIR}/cc-search" index
echo ""
"${INSTALL_DIR}/cc-search" status
echo ""
echo "Done! Use /cc-search:search-history in Claude Code to search your conversations."
```

- [ ] **Step 2: Update commands/search-history.md**

Replace the install block in `commands/search-history.md`:

```markdown
---
description: "Semantic search over past Claude Code conversations"
argument-hint: "QUERY [--all] [--top N]"
allowed-tools: ["Bash(cc-search:*)", "Bash(curl:*)", "Bash(wget:*)"]
---

Search past conversations using the `cc-search` CLI. Results include resume commands to jump back into any conversation.

IMPORTANT: Always display the FULL session ID (complete UUID) in results and resume commands. Never truncate or shorten session IDs — short hashes are useless for `claude --resume`.

```!
if ! command -v cc-search &>/dev/null; then
  OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
  ARCH="$(uname -m)"
  case "$ARCH" in x86_64) ARCH="amd64" ;; aarch64|arm64) ARCH="arm64" ;; esac
  BINARY="cc-search-${OS}-${ARCH}"
  URL="https://github.com/mikeler216/cc-search/releases/latest/download/${BINARY}"
  mkdir -p "$HOME/.local/bin"
  echo "Installing cc-search..." && curl -L -o "$HOME/.local/bin/cc-search" "$URL" && chmod +x "$HOME/.local/bin/cc-search"
  export PATH="$HOME/.local/bin:$PATH"
  echo "Building search index (first run)..." && cc-search index
fi
cc-search query $ARGUMENTS --top 5
```
```

- [ ] **Step 3: Update skills/search-history/SKILL.md install instructions**

In `skills/search-history/SKILL.md`, replace the install block under "Step 1":

```markdown
If the command fails (not found), install it:

```bash
# Detect platform and download binary
OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m)"
case "$ARCH" in x86_64) ARCH="amd64" ;; aarch64|arm64) ARCH="arm64" ;; esac
curl -L -o "$HOME/.local/bin/cc-search" "https://github.com/mikeler216/cc-search/releases/latest/download/cc-search-${OS}-${ARCH}"
chmod +x "$HOME/.local/bin/cc-search"
export PATH="$HOME/.local/bin:$PATH"
cc-search index
```
```

- [ ] **Step 4: Bump plugin.json version**

In `.claude-plugin/plugin.json`, update the version to `1.0.0` to mark the Go rewrite.

- [ ] **Step 5: Commit**

```bash
git add scripts/install.sh commands/search-history.md skills/search-history/SKILL.md .claude-plugin/plugin.json
git commit -m "feat: update plugin files for Go binary distribution"
```

---

### Task 10: CI/CD Workflows

**Files:**
- Modify: `.github/workflows/ci.yml`
- Create: `.github/workflows/release.yml`

Replace Python CI with Go build/test matrix, and add a release workflow for cross-platform binaries.

- [ ] **Step 1: Replace ci.yml with Go CI**

Replace `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    strategy:
      matrix:
        include:
          - os: ubuntu-latest
            arch: amd64
          - os: macos-latest
            arch: arm64

    runs-on: ${{ matrix.os }}

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-go@v5
        with:
          go-version: "1.22"

      - name: Install ONNX Runtime (Linux)
        if: runner.os == 'Linux'
        run: |
          wget -q https://github.com/microsoft/onnxruntime/releases/download/v1.17.1/onnxruntime-linux-x64-1.17.1.tgz
          tar xzf onnxruntime-linux-x64-1.17.1.tgz
          sudo cp onnxruntime-linux-x64-1.17.1/lib/* /usr/local/lib/
          sudo ldconfig

      - name: Install ONNX Runtime (macOS)
        if: runner.os == 'macOS'
        run: brew install onnxruntime

      - name: Export ONNX model
        run: |
          pip3 install optimum[exporters] transformers
          python3 scripts/export-model.py

      - name: Build
        run: CGO_ENABLED=1 go build ./cmd/cc-search

      - name: Test
        run: CGO_ENABLED=1 go test ./... -v -count=1 -timeout 300s

      - name: Vet
        run: go vet ./...
```

- [ ] **Step 2: Create release.yml**

Create `.github/workflows/release.yml`:

```yaml
name: Release

on:
  push:
    tags: ["v*"]

permissions:
  contents: write

jobs:
  build:
    strategy:
      matrix:
        include:
          - os: ubuntu-latest
            goos: linux
            goarch: amd64
            binary: cc-search-linux-amd64
          - os: macos-latest
            goos: darwin
            goarch: arm64
            binary: cc-search-darwin-arm64
          - os: macos-13
            goos: darwin
            goarch: amd64
            binary: cc-search-darwin-amd64

    runs-on: ${{ matrix.os }}

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-go@v5
        with:
          go-version: "1.22"

      - name: Install ONNX Runtime (Linux)
        if: runner.os == 'Linux'
        run: |
          wget -q https://github.com/microsoft/onnxruntime/releases/download/v1.17.1/onnxruntime-linux-x64-1.17.1.tgz
          tar xzf onnxruntime-linux-x64-1.17.1.tgz
          sudo cp onnxruntime-linux-x64-1.17.1/lib/* /usr/local/lib/
          sudo ldconfig

      - name: Install ONNX Runtime (macOS)
        if: runner.os == 'macOS'
        run: brew install onnxruntime

      - name: Export ONNX model
        run: |
          pip3 install optimum[exporters] transformers
          python3 scripts/export-model.py

      - name: Build
        env:
          CGO_ENABLED: "1"
          GOOS: ${{ matrix.goos }}
          GOARCH: ${{ matrix.goarch }}
        run: go build -ldflags="-s -w" -o ${{ matrix.binary }} ./cmd/cc-search

      - name: Generate checksum
        run: shasum -a 256 ${{ matrix.binary }} > ${{ matrix.binary }}.sha256

      - uses: actions/upload-artifact@v4
        with:
          name: ${{ matrix.binary }}
          path: |
            ${{ matrix.binary }}
            ${{ matrix.binary }}.sha256

  release:
    needs: build
    runs-on: ubuntu-latest

    steps:
      - uses: actions/download-artifact@v4
        with:
          path: artifacts
          merge-multiple: true

      - uses: softprops/action-gh-release@v2
        with:
          files: artifacts/*
          generate_release_notes: true
```

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/
git commit -m "feat: add Go CI and release workflows"
```

---

### Task 11: Cleanup — Remove Python Source

**Files:**
- Delete: `src/` (entire directory)
- Delete: `tests/` (entire directory)
- Delete: `pyproject.toml`
- Delete: `uv.lock`
- Delete: `.python-version`

Remove Python source, tests, and build config now that the Go implementation is complete.

- [ ] **Step 1: Verify Go binary works end-to-end**

```bash
CGO_ENABLED=1 go build -o cc-search ./cmd/cc-search
./cc-search index --claude-dir ~/.claude
./cc-search status
./cc-search query "test search" --all --top 3
```

Expected: all commands work, results are returned from real conversation history.

- [ ] **Step 2: Run Go test suite**

```bash
CGO_ENABLED=1 go test ./... -v -count=1 -timeout 300s
```

Expected: all tests pass.

- [ ] **Step 3: Remove Python files**

```bash
rm -rf src/ tests/ pyproject.toml uv.lock .python-version
```

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore: remove Python source after Go rewrite"
```
