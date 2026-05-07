package store

import (
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
	if err := db.InitMeta("test-hash"); err != nil {
		t.Fatalf("InitMeta: %v", err)
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
