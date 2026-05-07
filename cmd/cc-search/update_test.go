package main

import "testing"

func TestParseLatestReleaseTag(t *testing.T) {
	body := []byte(`{"tag_name":"v1.0.1"}`)

	tag, err := parseLatestReleaseTag(body)
	if err != nil {
		t.Fatalf("parseLatestReleaseTag: %v", err)
	}
	if tag != "v1.0.1" {
		t.Fatalf("tag = %q, want %q", tag, "v1.0.1")
	}
}

func TestSameVersionIgnoresLeadingV(t *testing.T) {
	if !sameVersion("1.0.1", "v1.0.1") {
		t.Fatal("expected versions with and without leading v to match")
	}
	if sameVersion("1.0.0", "v1.0.1") {
		t.Fatal("expected different versions not to match")
	}
}

func TestReleaseAssetURLUsesExplicitTag(t *testing.T) {
	got := releaseAssetURL("v1.0.1", "darwin", "amd64")
	want := "https://github.com/mikeler216/cc-search/releases/download/v1.0.1/cc-search-darwin-amd64"
	if got != want {
		t.Fatalf("releaseAssetURL() = %q, want %q", got, want)
	}
}
