package ingest

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
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
