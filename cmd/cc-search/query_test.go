package main

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/mikeler216/cc-search/internal/store"
)

func TestFilterRelevantResultsDropsWeakMatches(t *testing.T) {
	results := []store.Result{
		{Score: 0.37, ResumeCommand: "claude --resume weak-1"},
		{Score: 0.35, ResumeCommand: "claude --resume weak-2"},
	}

	filtered := filterRelevantResults(results)
	if len(filtered) != 0 {
		t.Fatalf("len(filtered) = %d, want 0", len(filtered))
	}
}

func TestFilterRelevantResultsKeepsStrongPrefix(t *testing.T) {
	results := []store.Result{
		{Score: 0.72, ResumeCommand: "claude --resume strong-1"},
		{Score: 0.58, ResumeCommand: "claude --resume strong-2"},
		{Score: 0.44, ResumeCommand: "claude --resume weak-3"},
	}

	filtered := filterRelevantResults(results)
	if len(filtered) != 2 {
		t.Fatalf("len(filtered) = %d, want 2", len(filtered))
	}
	if filtered[0].ResumeCommand != "claude --resume strong-1" {
		t.Fatalf("filtered[0] = %q", filtered[0].ResumeCommand)
	}
	if filtered[1].ResumeCommand != "claude --resume strong-2" {
		t.Fatalf("filtered[1] = %q", filtered[1].ResumeCommand)
	}
}

func TestFilterDisplayableResultsDropsMissingConversationFiles(t *testing.T) {
	existing := filepath.Join(t.TempDir(), "sess-1.jsonl")
	if err := os.WriteFile(existing, []byte("{}\n"), 0644); err != nil {
		t.Fatal(err)
	}
	subagent := filepath.Join(t.TempDir(), "subagents", "sess-2.jsonl")
	if err := os.MkdirAll(filepath.Dir(subagent), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(subagent, []byte("{}\n"), 0644); err != nil {
		t.Fatal(err)
	}

	results := []store.Result{
		{Score: 0.72, Chunk: store.Chunk{FilePath: existing, SessionID: "sess-1"}, ResumeCommand: "claude --resume sess-1"},
		{Score: 0.70, Chunk: store.Chunk{FilePath: filepath.Join(t.TempDir(), "missing.jsonl"), SessionID: "missing"}, ResumeCommand: "claude --resume missing"},
		{Score: 0.69, Chunk: store.Chunk{FilePath: existing, SessionID: ""}, ResumeCommand: "claude --resume "},
		{Score: 0.68, Chunk: store.Chunk{FilePath: subagent, SessionID: "sess-2"}, ResumeCommand: "claude --resume sess-2"},
		{
			Score: 0.67,
			Chunk: store.Chunk{
				FilePath:  existing,
				SessionID: "echo",
				Text:      "Search results for \"dev center\". Resume commands: claude --resume eafec706-0778-47bc-aff5-f74a86b94a39",
			},
			ResumeCommand: "claude --resume echo",
		},
	}

	filtered := filterDisplayableResults(results)
	if len(filtered) != 1 {
		t.Fatalf("len(filtered) = %d, want 1", len(filtered))
	}
	if filtered[0].SessionID != "sess-1" {
		t.Fatalf("SessionID = %q, want sess-1", filtered[0].SessionID)
	}
}
