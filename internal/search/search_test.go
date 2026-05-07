package search

import (
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
